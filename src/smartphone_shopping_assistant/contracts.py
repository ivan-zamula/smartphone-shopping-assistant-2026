"""Validated data contracts for search, intent parsing, and result presentation."""

from __future__ import annotations

from decimal import Decimal
from typing import Literal
from uuid import UUID

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator


class Contract(BaseModel):
    model_config = ConfigDict(extra="forbid")


class NumericRange(Contract):
    minimum: Decimal | None = None
    maximum: Decimal | None = None
    include_minimum: bool = True
    include_maximum: bool = True

    @model_validator(mode="after")
    def valid_range(self):
        if any(
            v is not None and isinstance(v, Decimal) and not v.is_finite()
            for v in (self.minimum, self.maximum)
        ):
            raise ValueError("range bounds must be finite")
        if any(v is not None and v < 0 for v in (self.minimum, self.maximum)):
            raise ValueError("price bounds must be nonnegative")
        if self.minimum is not None and self.maximum is not None:
            if self.minimum > self.maximum or (
                self.minimum == self.maximum and not (self.include_minimum and self.include_maximum)
            ):
                raise ValueError("range is empty or inverted")
        return self


class IntegerRange(NumericRange):
    minimum: int | None = None
    maximum: int | None = None

    @field_validator("minimum", "maximum", mode="before")
    @classmethod
    def strict_integer(cls, value):
        if value is not None and (isinstance(value, bool) or not isinstance(value, int)):
            raise ValueError("memory bounds must be integers")
        if value is not None and (value <= 0 or value > 2_147_483_647):
            raise ValueError("memory bounds must be positive")
        return value


class SearchFilters(Contract):
    brands: list[str] | None = None
    price: NumericRange | None = None
    ram_gb: IntegerRange | None = None
    storage_gb: IntegerRange | None = None
    carrier_unlocked: bool | None = Field(default=None, strict=True)

    @field_validator("brands")
    @classmethod
    def valid_brands(cls, value):
        if value is not None and (
            len(value) > 20 or any(not brand.strip() or len(brand) > 100 for brand in value)
        ):
            raise ValueError(
                "brands must contain at most 20 nonblank names of at most 100 characters"
            )
        return value


class FilterPatch(SearchFilters):
    """Unset fields mean unchanged; explicit null means clear the filter."""


class ProductCard(Contract):
    id: UUID
    name: str
    brand: str
    model: str
    ram_gb: int | None
    storage_gb: int | None
    color: str
    carrier_unlocked: bool = Field(strict=True)
    price: Decimal
    currency: str
    currency_status: str
    source_currency: str | None
    conversion_applied: bool
    raw_fields: dict[str, str]
    text: str

    @field_validator("ram_gb", "storage_gb", mode="before")
    @classmethod
    def positive_integer(cls, value):
        if value is not None and (
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
        ):
            raise ValueError("memory must be a positive integer or null")
        return value

    @field_validator("price")
    @classmethod
    def positive_price(cls, value):
        if not value.is_finite() or value <= 0:
            raise ValueError("price must be finite and positive")
        return value


class SearchRequest(Contract):
    query: str = Field(min_length=1, max_length=500)
    filters: SearchFilters = Field(default_factory=SearchFilters)
    limit: int = Field(default=5, ge=1, le=10, strict=True)
    mode: Literal["local"] = "local"

    @field_validator("query")
    @classmethod
    def nonblank(cls, value):
        if not value.strip():
            raise ValueError("query cannot be blank")
        return value.strip()


class SearchCandidate(Contract):
    product: ProductCard
    vector_score: float


class SearchResponse(Contract):
    request_id: UUID
    status: Literal["ok", "empty", "needs_clarification", "error"]
    filters: SearchFilters = Field(default_factory=SearchFilters)
    items: list[ProductCard] = Field(default_factory=list)
    message: str | None = None
    degraded: bool = False


class ParsedIntent(Contract):
    intent: Literal["search", "refine", "compare", "clarify"]
    semantic_query: str | None = Field(default=None, max_length=500)
    filter_changes: FilterPatch = Field(default_factory=FilterPatch)
    previous_result_positions: list[int] = Field(default_factory=list, max_length=10)

    @field_validator("previous_result_positions", mode="before")
    @classmethod
    def valid_positions(cls, value):
        if (
            not isinstance(value, list)
            or any(isinstance(i, bool) or not isinstance(i, int) or i <= 0 for i in value)
            or len(set(value)) != len(value)
        ):
            raise ValueError("positions must be unique positive integers")
        return value
