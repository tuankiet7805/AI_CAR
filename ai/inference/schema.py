"""Enum hư hỏng theo bảng yêu cầu; chưa phải schema API hoàn chỉnh."""

from enum import Enum


class DamageType(str, Enum):
    SCRATCH = "SCRATCH"
    DENT = "DENT"
    CRACK = "CRACK"
    # Hai enum này dành cho dữ liệu thực tế bổ sung sau, chưa có lớp CarDD tương ứng.
    TEAR_PUNCTURE = "TEAR_PUNCTURE"
    PAINT_CHIPPING = "PAINT_CHIPPING"
