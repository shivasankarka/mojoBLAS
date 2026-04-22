# ===----------------------------------------------------------------------=== #
# mojoBLAS: Mojo bindings for BLAS library
# Distributed under the MIT License.
# See LICENSE for more information.
#
# It is inspired by and based on the Netlib BLAS reference implementation:
# http://www.netlib.org/blas/
# ===----------------------------------------------------------------------=== #

"""
Modified Givens Rotation Constructor (`level1.rotmg`)
============================================

Provides modified Givens rotation construction as defined in the BLAS library standard.
"""


def rotmg[
    origin_d1: MutOrigin,
    origin_d2: MutOrigin,
    origin_x1: MutOrigin,
    mut_y1: Bool,
    origin_y1: Origin[mut=mut_y1],
    origin_param: MutOrigin,
    //,
    dtype: DType,
](
    d1: BLASPtr[dtype, origin_d1],
    d2: BLASPtr[dtype, origin_d2],
    x1: BLASPtr[dtype, origin_x1],
    y1: BLASPtr[dtype, origin_y1],
    param: BLASPtr[dtype, origin_param],
) -> None:
    comptime T = Scalar[dtype]
    var gam: T = 4096
    var gamsq: T = gam * gam
    var rgamsq: T = 1 / gamsq
    var zero: T = 0
    var one: T = 1
    var two: T = 2

    var dflag: T = -one
    var dh11: T = 0
    var dh12: T = 0
    var dh21: T = 0
    var dh22: T = 0

    var dd1 = d1[0]
    var dd2 = d2[0]
    var dx1 = x1[0]
    var dy1v = y1[0]

    if dd1 < zero:
        dflag = -one
        dd1 = zero
        dd2 = zero
        dx1 = zero
        dh11 = zero
        dh12 = zero
        dh21 = zero
        dh22 = zero
    else:
        var p2 = dd2 * dy1v
        if p2 == zero:
            param[0] = -two
            d1[0] = dd1
            d2[0] = dd2
            x1[0] = dx1
            return

        var p1 = dd1 * dx1
        var q2 = p2 * dy1v
        var q1 = p1 * dx1

        if abs(q1) > abs(q2):
            dh21 = -dy1v / dx1
            dh12 = p2 / p1
            var u = one - dh12 * dh21
            if u > zero:
                dflag = zero
                dd1 = dd1 / u
                dd2 = dd2 / u
                dx1 = dx1 * u
            else:
                dflag = -one
                dd1 = zero
                dd2 = zero
                dx1 = zero
                dh11 = zero
                dh12 = zero
                dh21 = zero
                dh22 = zero
        else:
            if q2 < zero:
                dflag = -one
                dd1 = zero
                dd2 = zero
                dx1 = zero
                dh11 = zero
                dh12 = zero
                dh21 = zero
                dh22 = zero
            else:
                dflag = one
                dh11 = p1 / p2
                dh22 = dx1 / dy1v
                var u = one + dh11 * dh22
                var temp = dd2 / u
                dd2 = dd1 / u
                dd1 = temp
                dx1 = dy1v * u

        if dd1 != zero:
            while dd1 <= rgamsq or dd1 >= gamsq:
                if dflag == zero:
                    dh11 = one
                    dh22 = one
                    dflag = -one
                else:
                    dh21 = -one
                    dh12 = one
                    dflag = -one

                if dd1 <= rgamsq:
                    dd1 = dd1 * gamsq
                    dx1 = dx1 / gam
                    dh11 = dh11 / gam
                    dh12 = dh12 / gam
                else:
                    dd1 = dd1 / gamsq
                    dx1 = dx1 * gam
                    dh11 = dh11 * gam
                    dh12 = dh12 * gam

        if dd2 != zero:
            while abs(dd2) <= rgamsq or abs(dd2) >= gamsq:
                if dflag == zero:
                    dh11 = one
                    dh22 = one
                    dflag = -one
                else:
                    dh21 = -one
                    dh12 = one
                    dflag = -one

                if abs(dd2) <= rgamsq:
                    dd2 = dd2 * gamsq
                    dh21 = dh21 / gam
                    dh22 = dh22 / gam
                else:
                    dd2 = dd2 / gamsq
                    dh21 = dh21 * gam
                    dh22 = dh22 * gam

    param[0] = dflag
    if dflag < zero:
        param[1] = dh11
        param[2] = dh21
        param[3] = dh12
        param[4] = dh22
    elif dflag == zero:
        param[2] = dh21
        param[3] = dh12
    else:
        param[1] = dh11
        param[4] = dh22

    d1[0] = dd1
    d2[0] = dd2
    x1[0] = dx1
