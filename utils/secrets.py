from dataclasses import dataclass, field

@dataclass(frozen=True)
class Secrets:
	url: str = ""
	key: str = ""