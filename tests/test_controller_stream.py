"""Controller stream regressions without a model, server, or controller thread."""

import json
import logging
import unittest
from unittest.mock import patch

from fastchat.constants import ErrorCode
# Importing the controller normally redirects stdout and creates a log file.
with patch("fastchat.utils.build_logger", return_value=logging.getLogger(__name__)):
    from fastchat.serve import controller as controller_module


class Response:
    def __init__(self, chunks=(), error=None, stream_error=None):
        self.chunks = chunks
        self.error = error
        self.stream_error = stream_error
        self.close_count = 0
        self.iterated = False

    def raise_for_status(self):
        if self.error is not None:
            raise self.error

    def iter_lines(self, **kwargs):
        self.iterated = True
        yield from self.chunks
        if self.stream_error is not None:
            raise self.stream_error

    def close(self):
        self.close_count += 1


class ControllerStreamTests(unittest.TestCase):
    def setUp(self):
        # Avoid starting the long-lived heartbeat thread.
        self.controller = controller_module.Controller.__new__(
            controller_module.Controller
        )
        self.params = {"model": "local-test", "prompt": "hello"}
        self.address = patch.object(
            self.controller, "get_worker_address", return_value="http://worker.invalid"
        )
        self.address.start()
        self.addCleanup(self.address.stop)

    def decoded(self, chunks):
        return [json.loads(chunk.rstrip(b"\0")) for chunk in chunks]

    def test_no_worker_yields_one_error_without_a_network_request(self):
        self.controller.get_worker_address.return_value = ""
        with patch.object(controller_module.requests, "post") as post:
            stream = self.controller.worker_api_generate_stream(self.params)
            first = next(stream)
            # Checking the first item alone misses the accidental second request.
            self.assertEqual(
                json.loads(first.rstrip(b"\0"))["error_code"],
                ErrorCode.CONTROLLER_NO_WORKER,
            )
            self.assertEqual(list(stream), [])
        post.assert_not_called()

    def test_success_preserves_null_delimited_chunks_and_closes(self):
        response = Response([b'{"text":"first"}', b"", b'{"text":"second"}'])
        with patch.object(
            controller_module.requests, "post", return_value=response
        ) as post:
            chunks = list(self.controller.worker_api_generate_stream(self.params))
        self.assertEqual(chunks, [b'{"text":"first"}\0', b'{"text":"second"}\0'])
        self.assertEqual(response.close_count, 1)
        post.assert_called_once_with(
            "http://worker.invalid/worker_generate_stream",
            json=self.params,
            stream=True,
            timeout=controller_module.WORKER_API_TIMEOUT,
        )

    def test_explicit_generator_close_releases_the_response(self):
        response = Response([b"first", b"second"])
        with patch.object(controller_module.requests, "post", return_value=response):
            stream = self.controller.worker_api_generate_stream(self.params)
            self.assertEqual(next(stream), b"first\0")
            stream.close()
        self.assertEqual(response.close_count, 1)

    def test_http_error_yields_timeout_without_forwarding_the_error_body(self):
        response = Response(
            [b"internal worker error"],
            error=controller_module.requests.exceptions.HTTPError("HTTP 503"),
        )
        with patch.object(controller_module.requests, "post", return_value=response):
            chunks = list(self.controller.worker_api_generate_stream(self.params))
        self.assertFalse(response.iterated)
        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            self.decoded(chunks)[0]["error_code"],
            ErrorCode.CONTROLLER_WORKER_TIMEOUT,
        )
        self.assertEqual(response.close_count, 1)

    def test_connection_failure_preserves_the_timeout_frame(self):
        error = controller_module.requests.exceptions.ConnectionError("unavailable")
        with patch.object(controller_module.requests, "post", side_effect=error):
            chunks = list(self.controller.worker_api_generate_stream(self.params))
        self.assertEqual(len(chunks), 1)
        self.assertEqual(
            self.decoded(chunks)[0]["error_code"],
            ErrorCode.CONTROLLER_WORKER_TIMEOUT,
        )

    def test_stream_failure_closes_after_preserving_prior_chunks(self):
        response = Response(
            [b'{"text":"partial"}'],
            stream_error=controller_module.requests.exceptions.ConnectionError("lost"),
        )
        with patch.object(controller_module.requests, "post", return_value=response):
            chunks = list(self.controller.worker_api_generate_stream(self.params))
        self.assertEqual(chunks[0], b'{"text":"partial"}\0')
        self.assertEqual(
            self.decoded(chunks[1:])[0]["error_code"],
            ErrorCode.CONTROLLER_WORKER_TIMEOUT,
        )
        self.assertEqual(response.close_count, 1)

    def test_unexpected_iterator_error_still_closes_and_propagates(self):
        response = Response(stream_error=ValueError("invalid worker chunk"))
        with patch.object(controller_module.requests, "post", return_value=response):
            with self.assertRaisesRegex(ValueError, "invalid worker chunk"):
                list(self.controller.worker_api_generate_stream(self.params))
        self.assertEqual(response.close_count, 1)


if __name__ == "__main__":
    unittest.main()
