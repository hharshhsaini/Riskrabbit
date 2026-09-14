from collections.abc import Generator
from typing import TYPE_CHECKING

from sqlalchemy import create_engine, select
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config import settings

if TYPE_CHECKING:
    from models import User

engine = create_engine(settings.DATABASE_URL, pool_pre_ping=True)

SessionLocal = sessionmaker(bind=engine, autocommit=False, autoflush=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def seed_default_user(db: Session) -> None:
    """Create the user named by DEFAULT_USER_EMAIL if it does not exist yet."""
    from models import User

    if db.execute(select(User.id).where(User.email == settings.DEFAULT_USER_EMAIL)).first() is None:
        db.add(User(email=settings.DEFAULT_USER_EMAIL))
        db.commit()


def get_default_user(db: Session) -> "User":
    """Return the single seeded user. Raises if startup seeding has not run."""
    from models import User

    user = db.execute(
        select(User).where(User.email == settings.DEFAULT_USER_EMAIL)
    ).scalar_one_or_none()
    if user is None:
        raise RuntimeError(
            f"No user found for DEFAULT_USER_EMAIL={settings.DEFAULT_USER_EMAIL!r}. "
            "The application startup handler seeds this row — check that it ran."
        )
    return user
