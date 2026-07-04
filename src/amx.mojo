# ===----------------------------------------------------------------------=== #
# Copyright (c) 2026, Modular Inc. All rights reserved.
#
# Licensed under the Apache License v2.0 with LLVM Exceptions:
# https://llvm.org/LICENSE.txt
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ===----------------------------------------------------------------------=== #
#
# This file contains wrappers around Apple's AMX assembly instruction set.
# For information on the Apple AMX instruction set, see
# https://www.notion.so/modularai/Apple-AMX-Resources-2cc523b9c851498787dfloat946ebb09930e.
#
# ===-----------------------------------------------------------------------===#

from std.sys._assembly import inlined_assembly

from std.memory import (
    memcpy,
    memset_zero,
    stack_allocation,
)


# All AMX instructions are of the form
# `0x00201000 | ((op & 0x1F) << 5) | (operand & 0x1F)`
# where `op` is the operation and `operand` is the register to operate on.


@always_inline
def _no_op_imms[op: Int32, imm: Int32]():
    # In Apple's Accelerate, instruction 17 is apparently always prefixed by
    # three nops.
    inlined_assembly[
        "nop\nnop\nnop\n.word (0x201000 + ($0 << 5) + $1)",
        NoneType,
        constraints="i,i,~{memory}",
        has_side_effect=True,
    ](op, imm)


@always_inline
def _op_gpr[op: Int32](gpr: Int64):
    inlined_assembly[
        ".word (0x201000 + ($0 << 5) + 0$1 - ((0$1 >> 4) * 6))",
        NoneType,
        constraints="i,r,~{memory}",
        has_side_effect=True,
    ](op, gpr)


# The `set` and `clr` take no non-constant operands, and so we pass them as
# immediate values via meta parameters.
@always_inline
def _set():
    _no_op_imms[17, 0]()


@always_inline
def _clr():
    _no_op_imms[17, 1]()


@always_inline
def ldx(gpr: Int):
    _op_gpr[0](Int64(gpr))


@always_inline
def ldy(gpr: Int):
    _op_gpr[1](Int64(gpr))


@always_inline
def stx(gpr: Int):
    _op_gpr[2](Int64(gpr))


@always_inline
def sty(gpr: Int):
    _op_gpr[3](Int64(gpr))


@always_inline
def ldz(gpr: Int):
    _op_gpr[4](Int64(gpr))


@always_inline
def stz(gpr: Int):
    _op_gpr[5](Int64(gpr))


@always_inline
def ldzi(gpr: Int):
    _op_gpr[6](Int64(gpr))


@always_inline
def stzi(gpr: Int):
    _op_gpr[7](Int64(gpr))


@always_inline
def extrx(gpr: Int):
    """
    Extracts a row or moves it to x, result in amx0.
    """
    _op_gpr[8](Int64(gpr))


@always_inline
def extry(gpr: Int):
    """
    Extracts a row or moves it to y, result in amx0.
    """
    _op_gpr[9](Int64(gpr))


@always_inline
def fma64(gpr: Int):
    """
    Float64 matrix multiply and add.
    """
    _op_gpr[10](Int64(gpr))


@always_inline
def fsm64(gpr: Int):
    """
    Float64 matrix multiply and subtract.
    """
    _op_gpr[11](Int64(gpr))


@always_inline
def fma32(gpr: Int):
    """
    Float32 matrix multiply and add.
    """
    _op_gpr[12](Int64(gpr))


@always_inline
def fsm32(gpr: Int):
    """
    Float32 matrix multiply and subtract.
    """
    _op_gpr[13](Int64(gpr))


@always_inline
def mac16(gpr: Int):
    """
    SI16 matrix multiply and add.
    """
    _op_gpr[14](Int64(gpr))


@always_inline
def fma16(gpr: Int):
    """
    Float16 matrix multiply and subtract.
    """
    _op_gpr[15](Int64(gpr))


@always_inline
def fms16(gpr: Int):
    """
    Float16 matrix multiply and add.
    """
    _op_gpr[16](Int64(gpr))


@always_inline
def vec_int__(gpr: Int):
    """
    Horizontal ui16 multiply `z0[i] += x0[i] + y0[i]`.
    """
    _op_gpr[18](Int64(gpr))


@always_inline
def vecfp(gpr: Int):
    """
    Horizontal float16 multiply `z0[i] += x0[i] + y0[i]`.
    """
    _op_gpr[19](Int64(gpr))


@always_inline
def max_int__(gpr: Int):
    """
    UI16 matrix multiply.
    """
    _op_gpr[20](Int64(gpr))


@always_inline
def matfp(gpr: Int):
    """
    Float16 matrix multiply.
    """
    _op_gpr[21](Int64(gpr))


@always_inline
def genlut(gpr: Int):
    _op_gpr[22](Int64(gpr))


# ===-----------------------------------------------------------------------===#
# Full operand encodings for the AMX instructions, reverse-engineered by
# corsix (https://github.com/corsix/amx). Bit positions below are referenced
# from that project's per-instruction docs (fma.md, fms.md, mac16.md,
# vecint.md, vecfp.md, matint.md, matfp.md, extr_x/y/h/v.md, genlut.md).
# These wrappers expose the *raw* bitfields of each instruction; callers are
# expected to know the legal combinations for their target dtype/mode.
# ===-----------------------------------------------------------------------===#

comptime AMX_VECTOR_MODE: Int = 1 << 63
comptime AMX_SKIP_X: Int = 1 << 29
comptime AMX_SKIP_Y: Int = 1 << 28
comptime AMX_SKIP_Z: Int = 1 << 27


@always_inline
def _fma_fms_operand[
    is_vector_mode: Bool
](
    x_offset: Int,
    y_offset: Int,
    z_row: Int,
    x_enable_mode: Int,
    x_enable_value: Int,
    y_enable_mode: Int,
    y_enable_value: Int,
    skip_x: Bool,
    skip_y: Bool,
    skip_z: Bool,
    z_is_f32: Bool,
    x_is_f16: Bool,
    y_is_f16: Bool,
) -> Int:
    """
    Shared bitfield layout for fma64/fms64/fma32/fms32/fma16/fms16.

    Bit 63: vector (1) or matrix (0) mode.
    Bit 62: Z is f32 (fma16 matrix mode only).
    Bit 61: X is f16 (fma32 only).
    Bit 60: Y is f16 (fma32 only).
    Bits 46:41 / 37:32: X / Y enable mode + value.
    Bits 29/28/27: skip X / skip Y / skip Z input.
    Bits 25:20: Z row.
    Bits 18:10: X offset (bytes). Bits 8:0: Y offset (bytes).
    """
    var operand = (
        (y_offset & 0x1FF) | ((x_offset & 0x1FF) << 10) | ((z_row & 0x3F) << 20)
    )

    comptime if is_vector_mode:
        operand |= AMX_VECTOR_MODE

    if skip_x:
        operand |= AMX_SKIP_X
    if skip_y:
        operand |= AMX_SKIP_Y
    if skip_z:
        operand |= AMX_SKIP_Z
    if z_is_f32:
        operand |= 1 << 62
    if x_is_f16:
        operand |= 1 << 61
    if y_is_f16:
        operand |= 1 << 60

    operand |= (x_enable_mode & 0x3) << 46
    operand |= (x_enable_value & 0x1F) << 41
    operand |= (y_enable_mode & 0x3) << 37
    operand |= (y_enable_value & 0x1F) << 32

    return operand


@always_inline
def fma_matrix[
    dtype: DType
](
    z_row: Int,
    x_offset: Int,
    y_offset: Int,
    x_enable_mode: Int = 0,
    x_enable_value: Int = 0,
    y_enable_mode: Int = 0,
    y_enable_value: Int = 0,
    skip_x: Bool = False,
    skip_y: Bool = False,
    skip_z: Bool = False,
    z_is_f32: Bool = False,
    x_is_f16: Bool = False,
    y_is_f16: Bool = False,
):
    """
    Matrix-mode (outer product) fma: z[j][i] += x[i] * y[j].

    `z_row` selects which Z row group accumulates the outer product (it is
    stepped by the underlying hardware: every 4th row for fma32/fms32, every
    2nd for fma16/fms16, every 8th for fma64/fms64 — see fma.md "Mode" table).
    `x_offset`/`y_offset` are byte offsets into the X/Y register pool (not
    register indices) and may span register boundaries.
    """
    comptime assert (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.float16
    ), "fma_matrix requires float16, float32, or float64"

    var operand = _fma_fms_operand[is_vector_mode=False](
        x_offset,
        y_offset,
        z_row,
        x_enable_mode,
        x_enable_value,
        y_enable_mode,
        y_enable_value,
        skip_x,
        skip_y,
        skip_z,
        z_is_f32,
        x_is_f16,
        y_is_f16,
    )

    comptime if dtype == DType.float64:
        fma64(operand)
    elif dtype == DType.float32:
        fma32(operand)
    else:
        fma16(operand)


@always_inline
def fma_vector[
    dtype: DType
](
    z_row: Int,
    x_offset: Int,
    y_offset: Int,
    enable_mode: Int = 0,
    enable_value: Int = 0,
    skip_x: Bool = False,
    skip_y: Bool = False,
    skip_z: Bool = False,
    x_is_f16: Bool = False,
    y_is_f16: Bool = False,
):
    """
    Vector-mode (pointwise) fma: z[_][i] += x[i] * y[i], writing one Z row.
    """
    comptime assert (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.float16
    ), "fma_vector requires float16, float32, or float64"

    var operand = _fma_fms_operand[is_vector_mode=True](
        x_offset,
        y_offset,
        z_row,
        enable_mode,
        enable_value,
        0,
        0,
        skip_x,
        skip_y,
        skip_z,
        False,
        x_is_f16,
        y_is_f16,
    )

    comptime if dtype == DType.float64:
        fma64(operand)
    elif dtype == DType.float32:
        fma32(operand)
    else:
        fma16(operand)


@always_inline
def fms_matrix[
    dtype: DType
](
    z_row: Int,
    x_offset: Int,
    y_offset: Int,
    x_enable_mode: Int = 0,
    x_enable_value: Int = 0,
    y_enable_mode: Int = 0,
    y_enable_value: Int = 0,
    skip_x: Bool = False,
    skip_y: Bool = False,
    skip_z: Bool = False,
    z_is_f32: Bool = False,
    x_is_f16: Bool = False,
    y_is_f16: Bool = False,
):
    """
    Matrix-mode (outer product) fms: z[j][i] -= x[i] * y[j].
    """
    comptime assert (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.float16
    ), "fms_matrix requires float16, float32, or float64"

    var operand = _fma_fms_operand[is_vector_mode=False](
        x_offset,
        y_offset,
        z_row,
        x_enable_mode,
        x_enable_value,
        y_enable_mode,
        y_enable_value,
        skip_x,
        skip_y,
        skip_z,
        z_is_f32,
        x_is_f16,
        y_is_f16,
    )

    comptime if dtype == DType.float64:
        fsm64(operand)
    elif dtype == DType.float32:
        fsm32(operand)
    else:
        fms16(operand)


@always_inline
def fms_vector[
    dtype: DType
](
    z_row: Int,
    x_offset: Int,
    y_offset: Int,
    enable_mode: Int = 0,
    enable_value: Int = 0,
    skip_x: Bool = False,
    skip_y: Bool = False,
    skip_z: Bool = False,
    x_is_f16: Bool = False,
    y_is_f16: Bool = False,
):
    """
    Vector-mode (pointwise) fms: z[_][i] -= x[i] * y[i], writing one Z row.
    """
    comptime assert (
        dtype == DType.float64
        or dtype == DType.float32
        or dtype == DType.float16
    ), "fms_vector requires float16, float32, or float64"

    var operand = _fma_fms_operand[is_vector_mode=True](
        x_offset,
        y_offset,
        z_row,
        enable_mode,
        enable_value,
        0,
        0,
        skip_x,
        skip_y,
        skip_z,
        False,
        x_is_f16,
        y_is_f16,
    )

    comptime if dtype == DType.float64:
        fsm64(operand)
    elif dtype == DType.float32:
        fsm32(operand)
    else:
        fms16(operand)


@always_inline
def mac16_op(
    z_row: Int,
    x_offset: Int,
    y_offset: Int,
    is_vector_mode: Bool,
    z_is_i32: Bool = False,
    x_is_i8: Bool = False,
    y_is_i8: Bool = False,
    right_shift: Int = 0,
    x_enable_mode: Int = 0,
    x_enable_value: Int = 0,
    y_enable_mode: Int = 0,
    y_enable_value: Int = 0,
    skip_x: Bool = False,
    skip_y: Bool = False,
    skip_z: Bool = False,
):
    """
    Integer multiply-accumulate: z[j][i] += (x[i] * y[j]) >> right_shift
    (matrix mode) or z[_][i] += (x[i] * y[i]) >> right_shift (vector mode).
    See mac16.md for the dtype/width table selected by z_is_i32/x_is_i8/y_is_i8.
    """
    var operand = (
        (y_offset & 0x1FF) | ((x_offset & 0x1FF) << 10) | ((z_row & 0x3F) << 20)
    )

    if is_vector_mode:
        operand |= AMX_VECTOR_MODE
    if z_is_i32:
        operand |= 1 << 62
    if x_is_i8:
        operand |= 1 << 61
    if y_is_i8:
        operand |= 1 << 60
    if skip_x:
        operand |= AMX_SKIP_X
    if skip_y:
        operand |= AMX_SKIP_Y
    if skip_z:
        operand |= AMX_SKIP_Z

    operand |= (right_shift & 0x1F) << 55
    operand |= (x_enable_mode & 0x3) << 46
    operand |= (x_enable_value & 0x1F) << 41
    operand |= (y_enable_mode & 0x3) << 37
    operand |= (y_enable_value & 0x1F) << 32

    mac16(operand)


@always_inline
def vecint_op(
    z_row: Int,
    x_offset: Int,
    y_offset: Int,
    alu_mode: Int,
    lane_width_mode: Int,
    right_shift: Int = 0,
    x_signed: Bool = False,
    y_signed: Bool = False,
    x_shuffle: Int = 0,
    y_shuffle: Int = 0,
    enable_mode: Int = 0,
    enable_value: Int = 0,
    repeat_multiple: Bool = False,
    repeat_four: Bool = False,
):
    """
    `vecint` (opcode 18): pointwise integer ALU between X, Y, and Z, e.g.
    z[_][i] +/-= (x[i] * y[i]) >> right_shift. See vecint.md for the full
    `alu_mode` (bits 47:42, when not an indexed load) and `lane_width_mode`
    (bits 45:42) tables; legal combinations depend on both.
    """
    var operand = (
        (y_offset & 0x1FF) | ((x_offset & 0x1FF) << 10) | ((z_row & 0x3F) << 20)
    )

    if x_signed:
        operand |= 1 << 63
    if y_signed:
        operand |= 1 << 26

    operand |= (right_shift & 0x1F) << 58
    operand |= (alu_mode & 0x3F) << 47
    operand |= (lane_width_mode & 0xF) << 42
    operand |= (x_shuffle & 0x3) << 29
    operand |= (y_shuffle & 0x3) << 27

    if repeat_multiple:
        operand |= 1 << 31
        if repeat_four:
            operand |= 1 << 25
        operand |= (enable_mode & 0x7) << 35
        operand |= (enable_value & 0x7) << 32
    else:
        operand |= (enable_mode & 0x7) << 38
        operand |= (enable_value & 0x3F) << 32

    vec_int__(operand)


@always_inline
def vecfp_op(
    z_row: Int,
    x_offset: Int,
    y_offset: Int,
    alu_mode: Int,
    lane_width_mode: Int,
    x_shuffle: Int = 0,
    y_shuffle: Int = 0,
    enable_mode: Int = 0,
    enable_value: Int = 0,
    repeat_multiple: Bool = False,
    repeat_four: Bool = False,
):
    """
    `vecfp` (opcode 19): pointwise float ALU between X, Y, and Z, e.g.
    z[_][i] += x[i] * y[i] (alu_mode 0) or z[_][i] -= x[i] * y[i] (alu_mode 1).
    See vecfp.md for the full `alu_mode` and `lane_width_mode` tables.
    """
    var operand = (
        (y_offset & 0x1FF) | ((x_offset & 0x1FF) << 10) | ((z_row & 0x3F) << 20)
    )

    operand |= (alu_mode & 0x3F) << 47
    operand |= (lane_width_mode & 0xF) << 42
    operand |= (x_shuffle & 0x3) << 29
    operand |= (y_shuffle & 0x3) << 27

    if repeat_multiple:
        operand |= 1 << 31
        if repeat_four:
            operand |= 1 << 25
        operand |= (enable_mode & 0x7) << 32
    else:
        operand |= (enable_mode & 0x7) << 38
        operand |= (enable_value & 0x1F) << 32

    vecfp(operand)


@always_inline
def matint_op(
    z_row: Int,
    x_offset: Int,
    y_offset: Int,
    alu_mode: Int,
    lane_width_mode: Int,
    right_shift: Int = 0,
    x_signed: Bool = False,
    y_signed: Bool = False,
    x_shuffle: Int = 0,
    y_shuffle: Int = 0,
    enable_mode_is_y: Bool = False,
    enable_mode: Int = 0,
    enable_value: Int = 0,
):
    """
    `matint` (opcode 20): outer-product integer ALU writing a 2D grid of Z,
    e.g. z[j][i] += (x[i] * y[j]) >> right_shift. See matint.md for the
    `alu_mode` and `lane_width_mode` tables (note: `z_row` here is only 2
    bits — bits 21:20 — high bits are ignored in matrix mode).
    """
    var operand = (
        (y_offset & 0x1FF) | ((x_offset & 0x1FF) << 10) | ((z_row & 0x3) << 20)
    )

    if x_signed:
        operand |= 1 << 63
    if y_signed:
        operand |= 1 << 26
    if enable_mode_is_y:
        operand |= 1 << 25

    operand |= (right_shift & 0x1F) << 58
    operand |= (alu_mode & 0x3F) << 47
    operand |= (lane_width_mode & 0xF) << 42
    operand |= (x_shuffle & 0x3) << 29
    operand |= (y_shuffle & 0x3) << 27
    operand |= (enable_mode & 0x7) << 38
    operand |= (enable_value & 0x3F) << 32

    max_int__(operand)


@always_inline
def matfp_op(
    z_row: Int,
    x_offset: Int,
    y_offset: Int,
    alu_mode: Int,
    lane_width_mode: Int,
    x_shuffle: Int = 0,
    y_shuffle: Int = 0,
    x_enable_mode: Int = 0,
    x_enable_value: Int = 0,
    y_enable_mode: Int = 0,
    y_enable_value: Int = 0,
):
    """
    `matfp` (opcode 21): outer-product float ALU writing a 2D grid of Z,
    e.g. z[j][i] += x[i] * y[j] (alu_mode 0). `z_row` is 3 bits (bits 22:20).
    Note the Y enable field is split: mode lives at bits 25:23, value at
    bits 62:57 (see matfp.md).
    """
    var operand = (
        (y_offset & 0x1FF) | ((x_offset & 0x1FF) << 10) | ((z_row & 0x7) << 20)
    )

    operand |= (alu_mode & 0x3F) << 47
    operand |= (lane_width_mode & 0xF) << 42
    operand |= (x_shuffle & 0x3) << 29
    operand |= (y_shuffle & 0x3) << 27
    operand |= (x_enable_mode & 0x7) << 38
    operand |= (x_enable_value & 0x1F) << 32
    operand |= (y_enable_mode & 0x7) << 23
    operand |= (y_enable_value & 0x3F) << 57

    matfp(operand)


@always_inline
def extrx_copy(x_register: Int, y_register: Int):
    """`extrx` whole-register copy: X[x_register][:] = Y[y_register][:]."""
    var operand = (
        (1 << 27) | ((y_register & 0x7) << 20) | ((x_register & 0x7) << 16)
    )
    extrx(operand)


@always_inline
def extry_copy(x_register: Int, y_register: Int):
    """`extry` whole-register copy: Y[y_register][:] = X[x_register][:]."""
    var operand = (
        (1 << 27) | ((x_register & 0x7) << 20) | ((y_register & 0x7) << 6)
    )
    extry(operand)


@always_inline
def extrh_row(
    destination_offset: Int,
    z_row: Int,
    enable_mode: Int = 0,
    enable_value: Int = 0,
    to_y: Bool = False,
):
    """
    `extrh` simple mode (26=0): copies one Z row (same lane width as X/Y) to
    X, or transposed to Y. See extr_h.md "Operand bitfields when 26=0".
    """
    var operand = (destination_offset & 0x1FF) << 10 | ((z_row & 0x3F) << 20)
    operand |= (enable_mode & 0x3) << 46
    operand |= (enable_value & 0x1F) << 41
    if to_y:
        operand |= 1 << 10

    extrx(operand)


@always_inline
def extrv_col(
    destination_offset: Int,
    z_column: Int,
    enable_mode: Int = 0,
    enable_value: Int = 0,
):
    """
    `extrv` simple mode (26=0): copies one Z column (same lane width as X/Y)
    to Y. See extr_v.md "Operand bitfields when 26=0".
    """
    var operand = (destination_offset & 0x1FF) | ((z_column & 0x3F) << 20)
    operand |= (enable_mode & 0x3) << 37
    operand |= (enable_value & 0x1F) << 32

    extry(operand)


@always_inline
def genlut_generate(
    destination_register: Int,
    source_offset: Int,
    mode: Int,
    table_register: Int,
    table_from_y: Bool = False,
    destination_is_y: Bool = False,
    source_from_y: Bool = False,
):
    """
    `genlut` generate mode (mode <= 6): builds a densely-packed index vector
    by binary-searching `table_register` for each lane of the source vector,
    writing the result to X or Y. See genlut.md for the mode table (selects
    source dtype / index width / lane count).
    """
    var operand = (
        (source_offset & 0x1FF)
        | ((mode & 0xF) << 53)
        | ((table_register & 0x7) << 60)
        | ((destination_register & 0x7) << 20)
    )
    if table_from_y:
        operand |= 1 << 59
    if destination_is_y:
        operand |= 1 << 25
    if source_from_y:
        operand |= 1 << 10

    genlut(operand)


@always_inline
def genlut_lookup(
    destination_register: Int,
    source_offset: Int,
    mode: Int,
    table_register: Int,
    destination_is_z: Bool = False,
    destination_is_y: Bool = False,
    table_from_y: Bool = False,
    source_from_y: Bool = False,
):
    """
    `genlut` lookup mode (mode >= 7): expands a densely-packed index vector
    from the source into full-width elements taken from `table_register`,
    writing the result to X, Y, or Z. See genlut.md for the mode table.
    """
    var operand = (
        (source_offset & 0x1FF)
        | ((mode & 0xF) << 53)
        | ((table_register & 0x7) << 60)
    )
    if table_from_y:
        operand |= 1 << 59
    if source_from_y:
        operand |= 1 << 10

    if destination_is_z:
        operand |= (1 << 26) | ((destination_register & 0x3F) << 20)
    else:
        if destination_is_y:
            operand |= 1 << 25
        operand |= (destination_register & 0x7) << 20

    genlut(operand)


# Apple.amx.LoadStore is a set of utilities that are thin wrappers around
# the inline assembly calls, and they provide an easier interface to use
# the amx registers.
#
# The M1 AMX hardware has 3 dedicated register banks, in fp32 mode they
# can be described as:
#
#     float X[8][16], Y[8][16], Z[64][16];
#
#  All instructions reading and writing these AMX registers are memory
#  instructions. The ops defined here marks the direction into/out of amx
#  registers. e.g. :
#
#       load_store.store_x(ptr, idx),
#
#   will read a row of 16 fp32 elements from memory at `ptr`, and save the
#   data in X[idx][:].
#   while
#
#       load_store.load_x (ptr, idx),
#
#   is the opposite, taking X[idx][:] and write to the memory location `ptr`.


@always_inline
def _encode_load_store[
    row_count: Int, dtype: DType
](src: UnsafePointer[Scalar[dtype], _], start_index: Int) -> Int:
    """
    Utility to do the bit encoding for load and store ops.
    """
    var src_idx = Int(src) | (start_index << 56)

    comptime if row_count == 2:
        src_idx |= 1 << 62
    return src_idx


@always_inline
def store_x[
    row_count: Int, dtype: DType
](src: UnsafePointer[Scalar[dtype], _], start_index: Int):
    ldx(_encode_load_store[row_count, dtype](src, start_index))


@always_inline
def store_y[
    row_count: Int, dtype: DType
](src: UnsafePointer[Scalar[dtype], _], start_index: Int):
    ldy(_encode_load_store[row_count, dtype](src, start_index))


@always_inline
def store_z[
    row_count: Int, dtype: DType
](src: UnsafePointer[Scalar[dtype], _], start_index: Int):
    ldz(_encode_load_store[row_count, dtype](src, start_index))


@always_inline
def read_x[
    row_count: Int, dtype: DType
](src: UnsafePointer[mut=True, Scalar[dtype], _], start_index: Int):
    stx(_encode_load_store[row_count, dtype](src, start_index))


@always_inline
def read_y[
    row_count: Int, dtype: DType
](src: UnsafePointer[mut=True, Scalar[dtype], _], start_index: Int):
    sty(_encode_load_store[row_count, dtype](src, start_index))


@always_inline
def load_z[
    row_count: Int, dtype: DType
](src: UnsafePointer[mut=True, Scalar[dtype], _], start_index: Int):
    stz(_encode_load_store[row_count, dtype](src, start_index))


@always_inline
def transpose_z_to_x_or_y[
    destination: StaticString, dtype: DType
](z_col_index: Int, xy_row_index: Int, z_row_suboffset: Int):
    # transpose_z_to_x_or_y is a thin wrapper around the fp32 transpose mode of
    # the amx instruction `extry`. This instruction takes a (sub) column of
    # register Z (see description above), and transposes it into a row in either
    # register X or register Y.
    #
    # Note that each column of Z has 64 element but each row of X or Y has only
    # 16 elements. The slightly strange part of this instruction is that the
    # value written into X/Y is actually a downsample (i.e. one in every four)
    # result of a column of Z.
    #
    # The instruction takes 1 static parameter dest and 3 dynamic parameters:
    # z_col_index, xy_row_index, and z_row_suboffset.
    # dest can be either `X` or `Y`.
    # With the X,Y,Z data layout described as
    #
    #    float X[8][16], Y[8][16], Z[64][16];
    #
    #  This instruction essentially takes:
    #
    #    extracted_column [16] = Z[z_row_suboffset : 64 : 4][z_col_index]
    #
    # and writes extracted_column[16] to X/Y[xy_row_index][:].
    #  Legal ranges for the parameters:
    #    z_col_index needs to be 0-15,
    #    xy_row_index needs to be 0-7,
    #    z_row_suboffset needs to be 0-4.

    # The destination must be either "X" or "Y".
    comptime assert destination == "X" or destination == "Y"
    # The type must be Float32.
    comptime assert dtype == DType.float32

    # make the y offset field
    #  shift left by 6 to make this an offset in rows,
    #    in fp32 mode, there are 16 elements / 64 byte per row.
    #  The offset field has to be given in bytes.
    var offset = ((z_col_index << 2) | z_row_suboffset) << 20 | (
        xy_row_index << 6
    )

    comptime is_x_destination = destination == "X"

    var operand = offset | (
        0x8000000004004000 if is_x_destination else 0x8000000010004000
    )

    extry(operand)


@always_inline
def fma[
    mode: StaticString, dtype: DType
](z_row_index: Int, x_row_index: Int, y_row_index: Int, clear_z: Bool):
    # Apple.amx.fma abstracts the fma operation on the amx hardware. Two modes of
    #  fma operations are supported in this instruction, referred to here as
    #  `RowMode` and `TileMode`.
    # `RowMode` is elementwise fma, for each set of given indices, the instruction
    #  computes z[z_row_index][:] += X[x_row_index][:] * Y[y_row_index][:].
    # `TileMode` is matrix fma, each op computes an outer product of:
    #   Y[y_row_index][:] X X[x_row_index][:], (generating a 16x16 matrix)
    #   and the resulting matrix is accumulated into Z[z_row_index::step 4][:].
    #  When clear_z is true, the existing value in Z will be ignored instead of
    #   being accumulated.
    #
    # Issues fma.fp32 instruction to AMX.
    #  Required input range (behavior for out of range is undefined):
    #  z_row_index : [0, 8) in row mode, [0, 4) in tile mode.
    #  x_row_index, y_row_index : always in [0, 8).

    # The mode must be either "TILE" or "ROW".
    comptime assert mode == "TILE" or mode == "ROW"
    # The type must be Float32.
    comptime assert dtype == DType.float32

    comptime is_row_mode = mode == "ROW"

    var operand = (
        y_row_index << 6
        | x_row_index << 16
        | z_row_index << 20
        | ((1 << 27) if clear_z else 0)
        | ((1 << 63) if is_row_mode else 0)
    )

    fma32(operand)


@always_inline
def dot_at_b[
    mut_a: Bool,
    mut_b: Bool,
    origin_a: Origin[mut=mut_a],
    origin_b: Origin[mut=mut_b],
    origin_c: MutOrigin,
    //,
    dtype: DType,
](
    c: UnsafePointer[Scalar[dtype], origin_c],
    ldc: Int,
    a: UnsafePointer[Scalar[dtype], origin_a],
    lda: Int,
    b: UnsafePointer[Scalar[dtype], origin_b],
    ldb: Int,
):
    """Performs a matrix multiply C = A^T * B using Apple AMX instructions.

    Supports f32 (16x16) and f16 (32x32) tiles, with all matrices stored in
    row-major order. `lda`/`ldb`/`ldc` are the leading dimensions (row
    strides) of A/B/C; each must be at least the tile's row length (16 for
    f32, 32 for f16).

    Parameters:
        mut_a: Mutability of pointer a.
        mut_b: Mutability of pointer b.
        origin_a: Memory origin of a.
        origin_b: Memory origin of b.
        origin_c: Memory origin of c (mutable, input/output).
        dtype: Element data type; must be float32 or float16.

    Args:
        c: Pointer to the output tile C (input/output, accumulated into).
        ldc: Leading dimension (row stride) of C.
        a: Pointer to the input tile A.
        lda: Leading dimension (row stride) of A.
        b: Pointer to the input tile B.
        ldb: Leading dimension (row stride) of B.
    """
    comptime assert (
        dtype == DType.float32 or dtype == DType.float16
    ), "the buffer dtype must be float32 or float16"

    comptime tile_dim = 16 if dtype == DType.float32 else 32
    comptime num_elements = tile_dim * tile_dim

    # TODO: We can elide the copy if the data is already aligned and
    # contiguous (i.e. lda/ldb/ldc == tile_dim).
    var a_buffer = stack_allocation[
        num_elements, Scalar[dtype], alignment=128
    ]()
    var b_buffer = stack_allocation[
        num_elements, Scalar[dtype], alignment=128
    ]()
    var c_buffer = stack_allocation[
        num_elements, Scalar[dtype], alignment=128
    ]()

    for row in range(tile_dim):
        memcpy(
            dest=a_buffer + row * tile_dim,
            src=a + row * lda,
            count=tile_dim,
        )
        memcpy(
            dest=b_buffer + row * tile_dim,
            src=b + row * ldb,
            count=tile_dim,
        )
    memset_zero(c_buffer, num_elements)

    # _set() has the side effect of clearing the z tile
    _set()

    comptime if dtype == DType.float32:
        comptime for j in range(2):
            comptime for i in range(8):
                ldx((i << 56) | Int(b_buffer + (j * 8 + i) * tile_dim))
                ldy((i << 56) | Int(a_buffer + (j * 8 + i) * tile_dim))

            comptime for i in range(8):
                fma32((i << 6 << 10) | (i << 6))

        comptime for i in range(0, 64, 4):
            stz((i << 56) | Int(c_buffer + (i >> 2) * tile_dim))
    elif dtype == DType.float16:
        comptime for j in range(4):
            comptime for i in range(8):
                ldx((i << 56) | Int(b_buffer + (j * 8 + i) * tile_dim))
                ldy((i << 56) | Int(a_buffer + (j * 8 + i) * tile_dim))

            comptime for i in range(8):
                fma16((i << 6 << 10) | (i << 6))

        comptime for i in range(0, 64, 2):
            stz((i << 56) | Int(c_buffer + (i >> 1) * tile_dim))

    _clr()

    for row in range(tile_dim):
        memcpy(
            dest=c + row * ldc,
            src=c_buffer + row * tile_dim,
            count=tile_dim,
        )
