from pydantic import BaseModel, Field


class PermissionScope(BaseModel):
    """What the caller is entitled to read.

    Deliberately separate from `Question.roles`, which records *who asked*. Both derive
    from the same token claim today, but they answer different questions and will diverge:
    a scope is where site restrictions and classification levels belong, and adding them
    should not change what a Question is.
    """

    roles: list[str] = Field(default_factory=list)

    @classmethod
    def from_roles(cls, roles: list[str]) -> "PermissionScope":
        return cls(roles=list(roles))
