from app import app
import uvicorn

__all__ = ["app"]


def main() -> None:
	uvicorn.run("app:app", host="0.0.0.0", port=7860)


if __name__ == "__main__":
	main()
