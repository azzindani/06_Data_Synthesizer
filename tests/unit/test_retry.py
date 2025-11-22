"""Unit tests for retry utilities."""

import pytest
import time

from src.utils.retry import retry_with_backoff, retry_operation


class TestRetryWithBackoff:
    """Tests for retry_with_backoff decorator."""

    def test_success_on_first_try(self):
        """Test function succeeds on first try."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01)
        def always_succeeds():
            nonlocal call_count
            call_count += 1
            return "success"

        result = always_succeeds()
        assert result == "success"
        assert call_count == 1

    def test_success_after_retries(self):
        """Test function succeeds after retries."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01, exceptions=(ValueError,))
        def succeeds_on_third():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise ValueError("Not yet")
            return "success"

        result = succeeds_on_third()
        assert result == "success"
        assert call_count == 3

    def test_failure_after_max_attempts(self):
        """Test function fails after max attempts."""
        call_count = 0

        @retry_with_backoff(max_attempts=3, base_delay=0.01, exceptions=(RuntimeError,))
        def always_fails():
            nonlocal call_count
            call_count += 1
            raise RuntimeError("Always fails")

        with pytest.raises(RuntimeError):
            always_fails()

        assert call_count == 3

    def test_only_catches_specified_exceptions(self):
        """Test only specified exceptions trigger retry."""
        @retry_with_backoff(max_attempts=3, base_delay=0.01, exceptions=(ValueError,))
        def raises_type_error():
            raise TypeError("Not caught")

        with pytest.raises(TypeError):
            raises_type_error()


class TestRetryOperation:
    """Tests for retry_operation function."""

    def test_success_on_first_try(self):
        """Test operation succeeds on first try."""
        result = retry_operation(
            lambda: "success",
            max_attempts=3,
            base_delay=0.01
        )
        assert result == "success"

    def test_success_after_retries(self):
        """Test operation succeeds after retries."""
        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 2:
                raise IOError("Retry")
            return "success"

        result = retry_operation(
            operation,
            max_attempts=3,
            base_delay=0.01,
            exceptions=(IOError,)
        )

        assert result == "success"
        assert call_count == 2

    def test_callback_on_retry(self):
        """Test callback is called on retry."""
        retries = []

        def on_retry(attempt, error, delay):
            retries.append(attempt)

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 3:
                raise Exception("Retry")
            return "success"

        result = retry_operation(
            operation,
            max_attempts=3,
            base_delay=0.01,
            on_retry=on_retry
        )

        assert result == "success"
        assert len(retries) == 2
        assert retries == [1, 2]

    def test_exponential_backoff(self):
        """Test delays increase exponentially."""
        delays = []

        def on_retry(attempt, error, delay):
            delays.append(delay)

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 4:
                raise Exception("Retry")
            return "success"

        retry_operation(
            operation,
            max_attempts=4,
            base_delay=1.0,
            exponential_base=2.0,
            on_retry=on_retry
        )

        # Delays should be: 1, 2, 4
        assert delays[0] == 1.0
        assert delays[1] == 2.0
        assert delays[2] == 4.0

    def test_max_delay_cap(self):
        """Test delay is capped at max_delay."""
        delays = []

        def on_retry(attempt, error, delay):
            delays.append(delay)

        call_count = 0

        def operation():
            nonlocal call_count
            call_count += 1
            if call_count < 5:
                raise Exception("Retry")
            return "success"

        retry_operation(
            operation,
            max_attempts=5,
            base_delay=10.0,
            max_delay=20.0,
            exponential_base=2.0,
            on_retry=on_retry
        )

        # All delays should be capped at 20
        for delay in delays:
            assert delay <= 20.0
