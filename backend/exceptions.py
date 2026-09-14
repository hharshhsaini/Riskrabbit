class GitHubAPIError(Exception):
    def __init__(
        self, message: str, status: int | None = None, retry_after: float | None = None
    ) -> None:
        super().__init__(message)
        self.message = message
        self.status = status
        # Seconds until GitHub will accept requests again; set only for rate limits.
        self.retry_after = retry_after


class PredictionError(Exception):
    def __init__(self, message: str, status: int | None = None) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class NotFoundError(Exception):
    def __init__(self, message: str) -> None:
        super().__init__(message)
        self.message = message
