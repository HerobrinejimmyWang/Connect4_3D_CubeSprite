from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from numbers import Integral
from types import MappingProxyType
from typing import Iterable


RULE_SCHEMA_VERSION = 1
FEATURE_SCHEMA_VERSION = 1
FEATURE_DIM = 32


class VerticalWinMode(str, Enum):
    NORMAL = "normal"
    IGNORED = "ignored"
    ILLEGAL = "illegal"


class Layer0WinMode(str, Enum):
    NORMAL = "normal"
    IGNORED = "ignored"


class NoLegalPlacementMode(str, Enum):
    DRAW = "draw"
    LOSS = "loss"
    FORCED_PASS = "forced_pass"


@dataclass(frozen=True)
class RuleModifier:
    """A composable change to one or more independent rule axes."""

    p1_vertical_mode: VerticalWinMode | None = None
    p1_layer0_mode: Layer0WinMode | None = None
    no_legal_placement_mode: NoLegalPlacementMode | None = None
    name: str = ""


@dataclass(frozen=True)
class RuleSpec:
    """Immutable, versioned rule semantics used by configs and game engines."""

    rule_id: str
    rule_code: int
    p1_vertical_mode: VerticalWinMode = VerticalWinMode.NORMAL
    p1_layer0_mode: Layer0WinMode = Layer0WinMode.NORMAL
    no_legal_placement_mode: NoLegalPlacementMode = NoLegalPlacementMode.DRAW
    rule_version: int = RULE_SCHEMA_VERSION
    feature_schema_version: int = FEATURE_SCHEMA_VERSION
    components: tuple[str, ...] = ()
    d4_symmetry: bool = True

    def __post_init__(self) -> None:
        if not self.rule_id or not self.rule_id.isascii():
            raise ValueError("rule_id must be a non-empty ASCII identifier.")
        if isinstance(self.rule_code, bool) or not isinstance(self.rule_code, Integral):
            raise TypeError("rule_code must be an integer.")
        object.__setattr__(self, "rule_code", int(self.rule_code))
        if not (0 <= self.rule_code <= 65535):
            raise ValueError("rule_code must fit in uint16.")
        if isinstance(self.rule_version, bool) or not isinstance(self.rule_version, Integral):
            raise TypeError("rule_version must be an integer.")
        object.__setattr__(self, "rule_version", int(self.rule_version))
        if self.rule_version <= 0:
            raise ValueError("rule_version must be positive.")
        if not isinstance(self.p1_vertical_mode, VerticalWinMode):
            raise TypeError("p1_vertical_mode must be a VerticalWinMode.")
        if not isinstance(self.p1_layer0_mode, Layer0WinMode):
            raise TypeError("p1_layer0_mode must be a Layer0WinMode.")
        if not isinstance(self.no_legal_placement_mode, NoLegalPlacementMode):
            raise TypeError("no_legal_placement_mode must be a NoLegalPlacementMode.")
        if not isinstance(self.d4_symmetry, bool):
            raise TypeError("d4_symmetry must be boolean.")
        if self.feature_schema_version != FEATURE_SCHEMA_VERSION:
            raise ValueError(
                f"Unsupported feature schema version {self.feature_schema_version}; "
                f"expected {FEATURE_SCHEMA_VERSION}."
            )
        object.__setattr__(self, "components", tuple(self.components))

    def to_contract_dict(self) -> dict[str, object]:
        return {
            "rule_id": self.rule_id,
            "rule_code": self.rule_code,
            "rule_version": self.rule_version,
            "feature_schema_version": self.feature_schema_version,
            "p1_vertical_mode": self.p1_vertical_mode.value,
            "p1_layer0_mode": self.p1_layer0_mode.value,
            "no_legal_placement_mode": self.no_legal_placement_mode.value,
            "components": list(self.components),
            "d4_symmetry": self.d4_symmetry,
        }


def compose_rule(
    *,
    rule_id: str,
    rule_code: int,
    modifiers: Iterable[RuleModifier],
    base: RuleSpec | None = None,
    rule_version: int = RULE_SCHEMA_VERSION,
) -> RuleSpec:
    """Build a rule from orthogonal modifiers, rejecting semantic conflicts."""

    vertical = base.p1_vertical_mode if base is not None else VerticalWinMode.NORMAL
    layer0 = base.p1_layer0_mode if base is not None else Layer0WinMode.NORMAL
    no_legal = base.no_legal_placement_mode if base is not None else NoLegalPlacementMode.DRAW
    assigned: dict[str, Enum] = {}
    component_names: list[str] = list(base.components if base is not None else ())

    for modifier in modifiers:
        if modifier.name:
            component_names.append(modifier.name)
        for field_name in (
            "p1_vertical_mode",
            "p1_layer0_mode",
            "no_legal_placement_mode",
        ):
            value = getattr(modifier, field_name)
            if value is None:
                continue
            prior = assigned.get(field_name)
            if prior is not None and prior != value:
                raise ValueError(
                    f"Conflicting modifiers for {field_name}: {prior.value!r} and {value.value!r}."
                )
            assigned[field_name] = value
            if field_name == "p1_vertical_mode":
                vertical = value
            elif field_name == "p1_layer0_mode":
                layer0 = value
            else:
                no_legal = value

    return RuleSpec(
        rule_id=rule_id,
        rule_code=rule_code,
        p1_vertical_mode=vertical,
        p1_layer0_mode=layer0,
        no_legal_placement_mode=no_legal,
        rule_version=rule_version,
        components=tuple(dict.fromkeys(component_names)),
    )


class RuleFeatureSchema:
    """Stable explicit feature layout; rule codes are never model inputs."""

    version = FEATURE_SCHEMA_VERSION
    dimension = FEATURE_DIM

    @classmethod
    def encode(cls, spec: RuleSpec) -> tuple[float, ...]:
        if spec.feature_schema_version != cls.version:
            raise ValueError(
                f"Rule {spec.rule_id!r} uses feature schema {spec.feature_schema_version}, "
                f"but encoder supports {cls.version}."
            )
        features = [0.0] * cls.dimension
        vertical_index = {
            VerticalWinMode.NORMAL: 0,
            VerticalWinMode.IGNORED: 1,
            VerticalWinMode.ILLEGAL: 2,
        }[spec.p1_vertical_mode]
        layer0_index = {
            Layer0WinMode.NORMAL: 3,
            Layer0WinMode.IGNORED: 4,
        }[spec.p1_layer0_mode]
        no_legal_index = {
            NoLegalPlacementMode.DRAW: 5,
            NoLegalPlacementMode.LOSS: 6,
            NoLegalPlacementMode.FORCED_PASS: 7,
        }[spec.no_legal_placement_mode]
        features[vertical_index] = 1.0
        features[layer0_index] = 1.0
        features[no_legal_index] = 1.0
        return tuple(features)


class RuleRegistry:
    """Read-only mapping between stable rule IDs, compact codes, and specs."""

    __slots__ = ("_specs", "_by_id", "_by_code", "_registry_hash", "_sealed")

    def __init__(self, specs: Iterable[RuleSpec]) -> None:
        object.__setattr__(self, "_sealed", False)
        ordered = tuple(sorted(specs, key=lambda spec: (spec.rule_code, spec.rule_id)))
        if not ordered:
            raise ValueError("RuleRegistry requires at least one rule.")
        by_id: dict[str, RuleSpec] = {}
        by_code: dict[int, RuleSpec] = {}
        for spec in ordered:
            if spec.rule_id in by_id:
                raise ValueError(f"Duplicate rule_id {spec.rule_id!r}.")
            if spec.rule_code in by_code:
                raise ValueError(f"Duplicate rule_code {spec.rule_code}.")
            by_id[spec.rule_id] = spec
            by_code[spec.rule_code] = spec
        object.__setattr__(self, "_specs", ordered)
        object.__setattr__(self, "_by_id", MappingProxyType(by_id))
        object.__setattr__(self, "_by_code", MappingProxyType(by_code))
        payload = {
            "feature_dim": FEATURE_DIM,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "rules": [spec.to_contract_dict() for spec in ordered],
        }
        canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
        object.__setattr__(self, "_registry_hash", hashlib.sha256(canonical.encode("utf-8")).hexdigest())
        object.__setattr__(self, "_sealed", True)

    def __setattr__(self, name: str, value: object) -> None:
        if getattr(self, "_sealed", False):
            raise AttributeError("RuleRegistry is immutable; construct a new registry instead.")
        object.__setattr__(self, name, value)

    @property
    def specs(self) -> tuple[RuleSpec, ...]:
        return self._specs

    @property
    def registry_hash(self) -> str:
        return self._registry_hash

    def get(self, identifier: str | int | RuleSpec) -> RuleSpec:
        if isinstance(identifier, RuleSpec):
            return identifier
        try:
            if isinstance(identifier, str):
                return self._by_id[identifier]
            if isinstance(identifier, bool) or not isinstance(identifier, Integral):
                raise TypeError("Rule identifier must be a rule ID, integer code, or RuleSpec.")
            return self._by_code[int(identifier)]
        except KeyError as exc:
            raise KeyError(f"Unknown rule {identifier!r}.") from exc

    def features(self, identifier: str | int | RuleSpec) -> tuple[float, ...]:
        return RuleFeatureSchema.encode(self.get(identifier))

    def __reduce__(self) -> tuple[type["RuleRegistry"], tuple[tuple[RuleSpec, ...]]]:
        """Rebuild read-only lookup proxies in spawned DataLoader workers."""

        return type(self), (self._specs,)


P1_VERTICAL_IGNORED_MODIFIER = RuleModifier(
    p1_vertical_mode=VerticalWinMode.IGNORED,
    name="p1_vertical_ignored",
)
P1_VERTICAL_FORBIDDEN_MODIFIER = RuleModifier(
    p1_vertical_mode=VerticalWinMode.ILLEGAL,
    no_legal_placement_mode=NoLegalPlacementMode.FORCED_PASS,
    name="p1_vertical_forbidden",
)
P1_LAYER0_IGNORED_MODIFIER = RuleModifier(
    p1_layer0_mode=Layer0WinMode.IGNORED,
    name="p1_layer0_ignored",
)

CLASSIC_RULE = compose_rule(rule_id="classic", rule_code=0, modifiers=())
P1_VERTICAL_IGNORED_RULE = compose_rule(
    rule_id="p1_vertical_ignored",
    rule_code=1,
    modifiers=(P1_VERTICAL_IGNORED_MODIFIER,),
)
P1_VERTICAL_FORBIDDEN_RULE = compose_rule(
    rule_id="p1_vertical_forbidden",
    rule_code=2,
    modifiers=(P1_VERTICAL_FORBIDDEN_MODIFIER,),
)
P1_LAYER0_IGNORED_RULE = compose_rule(
    rule_id="p1_layer0_ignored",
    rule_code=3,
    modifiers=(P1_LAYER0_IGNORED_MODIFIER,),
)

# Short aliases are convenient in experiments, while persisted IDs remain descriptive.
RULE1 = P1_VERTICAL_IGNORED_RULE
RULE2 = P1_VERTICAL_FORBIDDEN_RULE
RULE3 = P1_LAYER0_IGNORED_RULE

DEFAULT_RULE_REGISTRY = RuleRegistry((CLASSIC_RULE, RULE1, RULE2, RULE3))
