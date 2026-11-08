from __future__ import annotations

from abc import ABC, abstractmethod
from typing import ClassVar, Generic, TypeVar, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, ValidationError

from pixelflow.exceptions import InvalidImageError, InvalidParametersError

ImageArray = npt.NDArray[np.uint8 | np.uint16]


class OperationParams(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


P_co = TypeVar("P_co", bound=OperationParams, covariant=True)


class ImageOperation(ABC, Generic[P_co]):
    operation_name: ClassVar[str] = ""
    params_model: ClassVar[type[OperationParams]] = OperationParams

    def __init__(
        self,
        params: OperationParams | dict[str, object] | None = None,
        line: int | None = None,
    ) -> None:
        self.params: P_co = self._validate_params(params, line=line)

    @classmethod
    def _validate_params(
        cls,
        raw_params: OperationParams | dict[str, object] | None,
        line: int | None,
    ) -> P_co:
        if isinstance(raw_params, cls.params_model):
            return cast(P_co, raw_params)
        data = {} if raw_params is None else raw_params
        if not isinstance(data, dict):
            raise InvalidParametersError(
                cls.operation_name or cls.__name__,
                "Parameters must be a mapping",
                line=line,
            )
        try:
            validated = cls.params_model.model_validate(data)
            return cast(P_co, validated)
        except ValidationError as exc:
            error_list: list[dict[str, object]] = [
                {"loc": list(err["loc"]), "msg": err["msg"], "type": err["type"]}
                for err in exc.errors()
            ]
            raise InvalidParametersError(
                cls.operation_name or cls.__name__,
                error_list,
                line=line,
            ) from exc

    def __call__(self, image: ImageArray) -> ImageArray:
        return self.apply(image)

    def apply(self, image: ImageArray) -> ImageArray:
        validate_image_array(image)
        input_copy: ImageArray = image.copy()
        result = self._execute(input_copy)
        validate_image_array(result)
        return result

    @abstractmethod
    def _execute(self, image: ImageArray) -> ImageArray:
        raise NotImplementedError


def validate_image_array(image: object) -> None:
    if not isinstance(image, np.ndarray):
        raise InvalidImageError("Image must be a numpy ndarray")
    if image.dtype not in (np.uint8, np.uint16):
        raise InvalidImageError(f"Unsupported image dtype: {image.dtype}")
    if image.ndim not in (2, 3):
        raise InvalidImageError(f"Unsupported image dimensions: {image.ndim}")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise InvalidImageError(f"Image has invalid zero dimensions: {image.shape}")
    if image.ndim == 3 and image.shape[2] not in (1, 3, 4):
        raise InvalidImageError(f"Unsupported channel count: {image.shape[2]}")
