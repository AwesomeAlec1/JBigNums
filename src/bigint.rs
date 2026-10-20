/*
 * Copyright (c) 1996, 2025, Oracle and/or its affiliates. All rights reserved.
 * DO NOT ALTER OR REMOVE COPYRIGHT NOTICES OR THIS FILE HEADER.
 *
 * This code is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License version 2 only, as
 * published by the Free Software Foundation.  Oracle designates this
 * particular file as subject to the "Classpath" exception as provided
 * by Oracle in the LICENSE file that accompanied this code.
 *
 * This code is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
 * version 2 for more details (a copy is included in the LICENSE file that
 * accompanied this code).
 *
 * You should have received a copy of the GNU General Public License version
 * 2 along with this work; if not, write to the Free Software Foundation,
 * Inc., 51 Franklin St, Fifth Floor, Boston, MA 02110-1301 USA.
 *
 * Please contact Oracle, 500 Oracle Parkway, Redwood Shores, CA 94065 USA
 * or visit www.oracle.com if you need additional information or have any
 * questions.
 */

/*
 * Portions Copyright (c) 1995 Colin Plumb. All rights reserved.
 *
 * Rust port Copyright (c) 2026 Alec Clawson. All rights reserved.
 */

use std::cmp::Ordering;
use std::fmt::{Debug, Formatter, LowerHex, UpperHex};
use std::ops::*;
use std::sync::OnceLock;

use anyhow::{anyhow, Result};

/// An immutable, arbitrary-precision integer. All operations behave as if
/// `BigIntegers` were represented in two's-complement notation (like Rust's
/// primitive integer types). `BigInteger` provides implementations for all of
/// Rust's primitive integer operators, and many useful math methods.
/// Additionally, `BigInteger` provides operations for modular arithmetic, GCD
/// calculation, primality testing, prime generation, bit manipulation,
/// and a few other miscellaneous operations.
///
/// Semantics of shift operations extend those of Rust's shift operators
/// to allow for negative shift distances. A right-shift with a negative
/// shift distance results in a left shift, and vice versa.
///
/// Semantics of bitwise logical operations exactly mimic those of Rust's
/// bitwise integer operators. The binary operators ([`BitAnd`](BitAnd),
/// [`BitOr`](BitOr), [`BitXor`](BitXor)) implicitly perform sign extension
/// on the shorter of the two operands before performing the operation.
///
/// `BigInteger` implements [`PartialOrd`](PartialOrd) for all primitive signed
/// *and* unsigned integer types, in addition to [`Ord`](Ord) for comparison
/// between `BigIntegers`.
///
/// Modular arithmetic operations are provided to compute residues, perform
/// exponentiation, and compute multiplicative inverses. These methods always
/// return a non-negative result, between `0` and `(modulus - 1)`,
/// inclusive.
///
/// Bit operations operate on a single bit of the two's-complement
/// representation of their operand. If necessary, the operand is sign-extended
/// so that it contains the designated bit. None of the single-bit
/// operations can produce a `BigInteger` with a different sign from the
/// `BigInteger` being operated on, as they affect only a single bit, and the
/// arbitrarily large abstraction provided by this class ensures that conceptually
/// there are infinitely many "virtual sign bits" preceding each `BigInteger`.
///
/// For the sake of brevity and clarity, pseudocode is used throughout the
/// descriptions of `BigInteger` methods. The pseudocode expression
/// `(i + j)` is shorthand for "a `BigInteger` whose value is
/// that of the `BigInteger` `i` plus that of the `BigInteger` `j`."
/// The pseudocode expression `(i == j)` is shorthand for
/// "`true` if and only if the `BigInteger` `i` represents the same
/// value as the `BigInteger` `j`."  Other pseudocode expressions are
/// interpreted similarly.
///
/// BigInteger must support values in the range
/// -2<sup>`i32::MAX`</sup> (exclusive) to
/// +2<sup>`i32::MAX`</sup> (exclusive)
/// and may support values outside of that range.
///
/// An [error](anyhow::Error) is returned when a `BigInteger`
/// constructor or method would generate a value outside the
/// supported range.
///
/// The range of probable prime values is limited and may be less than
/// the full supported positive range of `BigInteger`.
/// The range must be at least 1 to 2<sup>500,000,000</sup>.
///
/// ## API Notes
/// <span id="algorithmicComplexity">As `BigInteger` values are arbitrary precision
/// integers, the algorithmic complexity of the methods of this class varies and may
/// be superlinear in the size of the input. For example, a method like `into::<i32>()`
/// would be expected to run in _O_(1), that is constant time, since with the current
/// internal representation only a fixed-size component of the `BigInteger` needs to
/// be accessed to perform the conversion to `int`. In contrast, a method like `not()`
/// would be expected to run in _O_(_n_) time, where _n_ is the size of the `BigInteger`
/// in bits, that is, to run in time proportional to the size of the input.
/// For multiplying two `BigInteger` values of size _n_, a naive multiplication
/// algorithm would run in time _O_(_n<sup>2</sup>_) and theoretical results indicate a
/// multiplication algorithm for numbers using this category of representation must run
/// in <em>at least</em> _O_(_n_&nbsp;log&nbsp;_n_). Common multiplication algorithms
/// between the bounds of the naive and theoretical cases include the Karatsuba
/// multiplication (_O_(_n<sup>1.585</sup>_)) and 3-way Toom-Cook multiplication
/// (_O_(_n<sup>1.465</sup>_)).</span>
///
/// A particular implementation of [`Mul`](Mul) is free to switch
/// between different algorithms for different inputs, such as to improve actual running
/// time to produce the product by using simpler algorithms for smaller inputs even if
/// the simpler algorithm has a larger asymptotic complexity.
///
/// Operations may also allocate and compute on intermediate results, potentially those
/// allocations may be as large as in proportion to the running time of the algorithm.
///
/// Users of `BigInteger` concerned with bounding the running time or space of operations
/// can screen out `BigInteger` values above a chosen magnitude.
///
/// See also: [`BigDecimal`]()
///
/// > Original Authors:
/// > - Josh Bloch
/// > - Michael McCloskey
/// > - Alan Eliasen
/// > - Timothy Buktu
/// >
/// > Rust port by Alec Clawson
pub struct BigInteger
{
    signum: i32,
    mag: Box<[u32]>,
    info: Box<BigIntegerInfo>,
}

const MAX_MAG_LENGTH: usize = usize::MAX / (size_of::<usize>() * 8) + 1; // 1 << 58

impl BigInteger
{
    pub(crate) fn create(mag: Box<[u32]>, signum: i32) -> Self
    {
        Self { mag, signum, info: Box::default() }
    }

    pub fn zero() -> Self { Self::create([].into(), 0) }

    /// Translates a byte subarray containing the two's-complement binary
    /// representation of a `BigInteger` into a `BigInteger`. The subarray
    /// is specified via an offset into the array and a length. The subarray
    /// is assumed to be in _big-endian_ byte-order: the most significant
    /// byte is the element at index `off`. The `val` array is assumed to be
    /// unchanged for the duration of the constructor call.
    ///
    /// An `OutOfBoundsError` is returned if the length of the array
    /// `val` is non-zero and `off+len` is greater than the length of `val`.
    ///
    /// ## Params
    /// `val` - byte array containing a subarray which is the big-endian
    ///         two's-complement binary representation of a `BigInteger`.
    ///
    /// `off` - the start offset of the binary representation.
    ///
    /// `len` - the number of bytes to use.
    pub fn from_byte_slice(val: &[u8], off: usize, len: usize) -> Result<Self>
    {
        if val.len() == 0
        {
            return Err(anyhow!("Zero-length BigInteger"));
        }

        check_from_index_size(off, len, val.len())?;

        if len == 0
        {
            return Ok(Self::zero());
        }

        let b = val[off];

        // Check if first byte is "negative"
        let r: BigInteger = if b & 0x80 != 0
        {
            let mag = make_positive(b, val, off, len);

            Self::create(mag.into(), -1)
        }
        else
        {
            let mag = strip_leading_zero_bytes(b, val, off, len);
            let signum = if mag.len() == 0 { 0 } else { 1 };

            Self::create(mag.into(), signum)
        };

        match check_range(&r.mag)
        {
            Ok(_) => Ok(r),
            Err(e) => Err(anyhow!(e))
        }
    }

    pub fn parse(val: &str, radix: u32) -> Result<Self>
    {
        let radix_size = radix as usize;

        let mut cursor: usize = 0;

        let len = val.len();

        if radix < 2 || radix > 36
        {
            return Err(anyhow!("Radix out of range"));
        }

        if len == 0
        {
            return Err(anyhow!("Zero-length BigInteger"));
        }

        // Check for at most one leading sign
        let mut sign = 1;

        let dash_idx = val.rfind('-').unwrap_or(usize::MAX);
        let plus_idx = val.rfind('+').unwrap_or(usize::MAX);

        if dash_idx != usize::MAX
        {
            if dash_idx != 0 || plus_idx != usize::MAX
            {
                return Err(anyhow!("Illegal embedded sign character"));
            }
            sign = -1;
            cursor = 1;
        }
        else if plus_idx != usize::MAX
        {
            if plus_idx != 0
            {
                return Err(anyhow::anyhow!("Illegal embedded sign character"));
            }
            cursor = 1;
        }

        if cursor == len
        {
            return Err(anyhow::anyhow!("Zero-length BigInteger"));
        }

        // Skip leading zeros and compute number of digits in magnitude
        while cursor < len && val.chars()
                                 .nth(cursor)
                                 .unwrap_or(' ')
                                 .to_digit(radix)
                                 .unwrap_or(u32::MAX) == 0
        {
            cursor += 1;
        }

        if cursor == len
        {
            return Ok(Self::zero());
        }

        let num_digits = len - cursor;

        let signum: i32 = sign;

        // Pre-allocate array of expected size. May be too large but can
        // never be too small. Typically exact.
        let num_bits = ((num_digits * BITS_PER_DIGIT[radix_size] as usize) >> 10) + 1;

        if num_bits + 31 >= (1usize << 32)
        {
            return Err(anyhow!("BigInteger would overflow supported range"));
        }

        let num_words = (num_bits + 31) >> 5;

        let mut magnitude = vec![0; num_words];

        // Process first (potentially short) digit group
        let mut first_group_len = num_digits % DIGITS_PER_INT[radix_size] as usize;
        if first_group_len == 0
        {
            first_group_len = DIGITS_PER_INT[radix_size] as usize;
        }

        let mut group: String = val.chars()
                                   .skip(cursor)
                                   .take(first_group_len)
                                   .collect();

        cursor += 1;

        match u32::from_str_radix(&group, radix)
        {
            Ok(n) => {
                magnitude[num_words - 1] = n;
            }
            Err(_) => return Err(anyhow!("Illegal digit"))
        };

        // Process remaining digit groups
        let super_radix = INT_RADIX[radix_size];

        let mut group_val;

        while cursor < len
        {
            group = val.chars()
                       .skip(cursor)
                       .take(DIGITS_PER_INT[radix_size] as usize)
                       .collect();

            cursor += 1;

            group_val = u32::from_str_radix(&group, radix);

            match group_val
            {
                Ok(n) => {
                    destructive_mul_add(&mut magnitude, super_radix, n);
                }
                Err(_) => return Err(anyhow!("Illegal digit"))
            };
        }

        // Required for cases where the array was overallocated.
        let mag = trusted_strip_leading_zero_ints(magnitude);
        if mag.len() >= MAX_MAG_LENGTH
        {
            check_range(&mag)?;
        }
        Ok(Self::create(mag.into(), signum))
    }

    pub fn parallel_multiply(&self, _rhs: &Self) -> Self
    {
        todo!()
    }

    pub fn divide_and_remainder(self, _rhs: Self) -> (Self, Self)
    {
        todo!()
    }

    pub fn abs(mut self) -> Self
    {
        if self.signum < 0
        {
            self.signum = 1;
        }
        self
    }

    pub fn signum(&self) -> i32 { self.signum }

    /// Returns an immutable reference to the slice
    /// containing the integer array of this `BigInteger`.
    fn magnitude(&self) -> &[u32] { &self.mag }

    /// Returns the index of the rightmost (lowest-order) one bit in this
    /// `BigInteger` (the number of zero bits to the right of the rightmost
    /// one bit).  Returns `None` if this `BigInteger` contains no one-bits.
    /// (Computes `(this == 0? -1 : log2(this & -this))`.)
    pub fn lowest_set_bit(&self) -> Option<u64>
    {
        *self.info.lowest_set_bit.get_or_init(|| {
            if self.signum == 0
            {
                None
            }
            else
            {
                let tz = self.num_trailing_zero_ints();
                let lsb = tz * size_of::<u32>() as u64
                    + self.mag[self.mag.len() - 1 - tz as usize].trailing_zeros() as u64;

                Some(lsb)
            }
        })
    }

    /// Returns the number of bits in the two's complement representation
    /// of this `BigInteger` that differ from its sign bit. This method is
    /// useful when implementing bit-vector style sets atop `BigInteger`s.
    pub fn count_ones(&self) -> u64
    {
        *self.info.bit_cnt.get_or_init(|| {
            let mut bc = 0;

            let first_zero_int = self.mag.len() - self.num_trailing_zero_ints() as usize;

            let mut i = 0;
            while i < first_zero_int
            {
                bc += self.mag[i].count_ones() as u64;
                i += 1;
            }
            if self.signum < 0
            {
                bc += self.lowest_set_bit().unwrap() - 1;
            }
            bc
        })
    }

    fn sign_int(&self) -> u32
    {
        if self.signum < 0 { u32::MAX } else { 0 }
    }

    fn get_int(&self, idx: usize) -> u32
    {
        if idx >= self.mag.len()
        {
            self.sign_int()
        }
        else
        {
            let mag_int = self.mag[self.mag.len() - idx - 1];

            if self.signum >= 0
            {
                mag_int
            }
            else
            {
                if idx <= self.num_trailing_zero_ints() as usize
                {
                    !mag_int + 1
                }
                else
                {
                    !mag_int
                }
            }
        }
    }

    fn num_trailing_zero_ints(&self) -> u64
    {
        *self.info.num_trailing_zero_ints.get_or_init(|| {
            let lowest = self.mag.len() - 1;

            let mut i = lowest;
            let mut ovf = false;
            while !ovf && self.mag[i] == 0
            {
                (i, ovf) = i.overflowing_sub(1);
            }
            (lowest - i) as u64
        })
    }
}

impl Default for BigInteger
{
    fn default() -> Self { Self::zero() }
}

impl Clone for BigInteger
{
    fn clone(&self) -> Self
    {
        println!("BigInteger cloned");
        Self {
            mag:    self.mag.clone(),
            signum: self.signum,
            info:   self.info.clone()
        }
    }
}

impl From<u128> for BigInteger
{
    fn from(val: u128) -> Self
    {
        if val == 0
        {
            return Self::zero();
        }

        let d0 = (val >> 96) as u32;
        let d1 = (val >> 64) as u32;
        let d2 = (val >> 32) as u32;
        let d3 = val as u32;

        let arr = [d0, d1, d2, d3];

        let first_nonzero = arr.iter()
                               .position(|&x| x != 0)
                               .unwrap();

        Self::create(arr[first_nonzero..].into(), 1)
    }
}

macro_rules! impl_bigint_from_unsigned {
    ($($t:ty),*) => {
        $(
            impl From<$t> for BigInteger
            {
                fn from(val: $t) -> Self
                {
                    Self::from(val as u128)
                }
            }
        )*
    };
}

impl_bigint_from_unsigned!(u8, u16, u32, u64, usize);

impl From<i128> for BigInteger
{
    fn from(val: i128) -> Self
    {
        if val < 0
        {
            -Self::from(val.unsigned_abs())
        }
        else
        {
            Self::from(val.cast_unsigned())
        }
    }
}

macro_rules! impl_bigint_from_signed {
    ($($t:ty),*) => {
        $(
            impl From<$t> for BigInteger
            {
                fn from(val: $t) -> Self
                {
                    Self::from(val as i128)
                }
            }
        )*
    };
}

impl_bigint_from_signed!(i8, i16, i32, i64, isize);

impl TryFrom<&str> for BigInteger
{
    type Error = anyhow::Error;

    fn try_from(s: &str) -> Result<Self, Self::Error>
    {
        Self::parse(s, 10)
    }
}

impl TryFrom<&[u8]> for BigInteger
{
    type Error = anyhow::Error;

    fn try_from(val: &[u8]) -> Result<Self>
    {
        Self::from_byte_slice(val, 0, val.len())
    }
}

impl Add for BigInteger
{
    type Output = Self;

    fn add(self, rhs: Self) -> Self::Output
    {
        if self.signum == 0
        {
            rhs
        }
        else if rhs.signum == 0
        {
            self
        }
        else
        {
            &self + &rhs
        }
    }
}

impl Add for &BigInteger
{
    type Output = BigInteger;

    fn add(self, rhs: Self) -> Self::Output
    {
        if self.signum == 0
        {
            rhs.clone()
        }
        else if rhs.signum == 0
        {
            self.clone()
        }
        else if self.signum == rhs.signum
        {
            BigInteger::create(add_slices(&self.mag, &rhs.mag).into(), self.signum)
        }
        else
        {
            let cmp = cmp_slices(&self.mag, &rhs.mag);

            if cmp == Ordering::Equal
            {
                return BigInteger::zero();
            }

            let mut res = if cmp == Ordering::Greater
            {
                subtract_slices(&self.mag, &rhs.mag)
            }
            else
            {
                subtract_slices(&rhs.mag, &self.mag)
            };

            res = trusted_strip_leading_zero_ints(res);

            BigInteger::create(res.into(), get_signum_for_ord(cmp, self.signum))
        }
    }
}

impl Add<u128> for BigInteger
{
    type Output = Self;

    fn add(self, rhs: u128) -> Self::Output
    {
        if rhs == 0
        {
            self
        }
        else
        {
            // No moves to be had, we can just reuse the code for adding to &self.
            &self + rhs
        }
    }
}

impl Add<u128> for &BigInteger
{
    type Output = BigInteger;

    fn add(self, rhs: u128) -> Self::Output
    {
        if rhs == 0
        {
            self.clone()
        }
        else if self.signum == 0
        {
            BigInteger::from(rhs)
        }
        else if self.signum > 0
        {
            BigInteger::create(add_u128_to_slice(&self.mag, rhs).into(), self.signum)
        }
        else
        {
            let cmp = cmp_slice_to_u128(&self.mag, rhs);

            if cmp == Ordering::Equal
            {
                return BigInteger::zero();
            }

            let mut res = if cmp == Ordering::Greater
            {
                subtract_u128_from_slice(&self.mag, rhs)
            }
            else
            {
                subtract_slice_from_u128(rhs, &self.mag)
            };

            res = trusted_strip_leading_zero_ints(res);

            BigInteger::create(res.into(), get_signum_for_ord(cmp, self.signum))
        }
    }
}

macro_rules! impl_bigint_add_unsigned {
    ($($t:ty),*) => {
        $(
            impl Add<$t> for BigInteger
            {
                type Output = Self;

                fn add(self, rhs: $t) -> Self::Output
                {
                    self + rhs as u128
                }
            }

            impl Add<$t> for &BigInteger
            {
                type Output = BigInteger;

                fn add(self, rhs: $t) -> Self::Output
                {
                    self + rhs as u128
                }
            }
        )*
    };
}

impl_bigint_add_unsigned!(u8, u16, u32, u64, usize);

// For signed longs, we can just check if the rhs is negative, and if so,
// perform the opposite operation with the unsigned absolute value instead.
impl Add<i128> for BigInteger
{
    type Output = Self;

    fn add(self, rhs: i128) -> Self::Output
    {
        if rhs == 0
        {
            self
        }
        else if rhs < 0
        {
            self - rhs.unsigned_abs()
        }
        else
        {
            self + rhs.cast_unsigned()
        }
    }
}

impl Add<i128> for &BigInteger
{
    type Output = BigInteger;

    fn add(self, rhs: i128) -> Self::Output
    {
        if rhs == 0
        {
            self.clone()
        }
        else if rhs < 0
        {
            self - rhs.unsigned_abs()
        }
        else
        {
            self + rhs.cast_unsigned()
        }
    }
}

macro_rules! impl_bigint_add_signed {
    ($($t:ty),*) => {
        $(
            impl Add<$t> for BigInteger
            {
                type Output = Self;

                fn add(self, rhs: $t) -> Self::Output
                {
                    self + rhs as i128
                }
            }

            impl Add<$t> for &BigInteger
            {
                type Output = BigInteger;

                fn add(self, rhs: $t) -> Self::Output
                {
                    self + rhs as i128
                }
            }
        )*
    };
}

impl_bigint_add_signed!(i8, i16, i32, i64, isize);

impl AddAssign for BigInteger
{
    fn add_assign(&mut self, rhs: Self)
    {
        if self.signum == 0
        {
            *self = rhs;
        }
        else if rhs.signum != 0
        {
            *self = &*self + &rhs;
        }
        // If rhs == 0, do nothing
    }
}

impl AddAssign<&BigInteger> for BigInteger
{
    fn add_assign(&mut self, rhs: &BigInteger)
    {
        *self = &*self + rhs
    }
}

impl AddAssign<u128> for BigInteger
{
    fn add_assign(&mut self, rhs: u128)
    {
        if self.signum == 0
        {
            *self = BigInteger::from(rhs);
        }
        else if rhs == 0
        {
            return;
        }
        else
        {
            *self = &*self + rhs;
        }
    }
}

macro_rules! impl_bigint_add_assign_unsigned {
    ($($t:ty),*) => {
        $(
            impl AddAssign<$t> for BigInteger
            {
                fn add_assign(&mut self, rhs: $t)
                {
                    self.add_assign(rhs as u128)
                }
            }
        )*
    };
}

impl_bigint_add_assign_unsigned!(u8, u16, u32, u64, usize);

impl AddAssign<i128> for BigInteger
{
    fn add_assign(&mut self, rhs: i128)
    {
        if self.signum == 0
        {
            *self = BigInteger::from(rhs);
        }
        else if rhs == 0
        {
            return;
        }
        else if rhs < 0
        {
            *self = &*self - rhs.unsigned_abs();
        }
        else
        {
            *self = &*self + rhs;
        }
    }
}

macro_rules! impl_bigint_add_assign_signed {
    ($($t:ty),*) => {
        $(
            impl AddAssign<$t> for BigInteger
            {
                fn add_assign(&mut self, rhs: $t)
                {
                    self.add_assign(rhs as i128)
                }
            }
        )*
    };
}

impl_bigint_add_assign_signed!(i8, i16, i32, i64, isize);

impl Sub for BigInteger
{
    type Output = Self;

    fn sub(self, rhs: Self) -> Self::Output
    {
        if rhs.signum == 0
        {
            self
        }
        else if self.signum == 0
        {
            rhs
        }
        else
        {
            &self - &rhs
        }
    }
}

impl Sub for &BigInteger
{
    type Output = BigInteger;

    fn sub(self, rhs: Self) -> Self::Output
    {
        if rhs.signum == 0
        {
            self.clone()
        }
        else if self.signum == 0
        {
            rhs.clone()
        }
        else
        {
            if self.signum != rhs.signum
            {
                BigInteger::create(add_slices(&self.mag, &rhs.mag).into(), self.signum)
            }
            else
            {
                let cmp = cmp_slices(&self.mag, &rhs.mag);

                if cmp == Ordering::Equal { return BigInteger::zero(); }

                let mut res = if cmp == Ordering::Greater
                {
                    subtract_slices(&self.mag, &rhs.mag)
                }
                else
                {
                    subtract_slices(&rhs.mag, &self.mag)
                };

                res = trusted_strip_leading_zero_ints(res);

                BigInteger::create(res.into(), get_signum_for_ord(cmp, self.signum))
            }
        }
    }
}

impl Sub<u128> for BigInteger
{
    type Output = Self;

    fn sub(self, rhs: u128) -> Self::Output
    {
        if rhs == 0
        {
            self
        }
        else
        {
            &self - rhs
        }
    }
}

impl Sub<u128> for &BigInteger
{
    type Output = BigInteger;

    fn sub(self, rhs: u128) -> Self::Output
    {
        if rhs == 0
        {
            self.clone()
        }
        else if self.signum == 0
        {
            -BigInteger::from(rhs)
        }
        else if self.signum < 0
        {
            BigInteger::create(add_u128_to_slice(&self.mag, rhs).into(), self.signum)
        }
        else
        {
            let cmp = cmp_slice_to_u128(&self.mag, rhs);

            if cmp == Ordering::Equal
            {
                return BigInteger::zero();
            }

            let mut res = if cmp == Ordering::Greater
            {
                subtract_u128_from_slice(&self.mag, rhs)
            }
            else
            {
                subtract_slice_from_u128(rhs, &self.mag)
            };

            res = trusted_strip_leading_zero_ints(res);

            BigInteger::create(res.into(), get_signum_for_ord(cmp, self.signum))
        }
    }
}

macro_rules! impl_bigint_sub_unsigned {
    ($($t:ty),*) => {
        $(
            impl Sub<$t> for BigInteger
            {
                type Output = Self;

                fn sub(self, rhs: $t) -> Self::Output
                {
                    self - rhs as u128
                }
            }

            impl Sub<$t> for &BigInteger
            {
                type Output = BigInteger;

                fn sub(self, rhs: $t) -> Self::Output
                {
                    self - rhs as u128
                }
            }
        )*
    };
}

impl_bigint_sub_unsigned!(u8, u16, u32, u64, usize);

impl Sub<i128> for BigInteger
{
    type Output = Self;

    fn sub(self, rhs: i128) -> Self::Output
    {
        if rhs == 0
        {
            self
        }
        else if rhs < 0
        {
            self + rhs.unsigned_abs()
        }
        else
        {
            self - rhs.cast_unsigned()
        }
    }
}

impl Sub<i128> for &BigInteger
{
    type Output = BigInteger;

    fn sub(self, rhs: i128) -> Self::Output
    {
        if rhs == 0
        {
            self.clone()
        }
        else if rhs < 0
        {
            self + rhs.unsigned_abs()
        }
        else
        {
            self - rhs.cast_unsigned()
        }
    }
}

macro_rules! impl_bigint_sub_signed {
    ($($t:ty),*) => {
        $(
            impl Sub<$t> for BigInteger
            {
                type Output = Self;

                fn sub(self, rhs: $t) -> Self::Output
                {
                    self - rhs as i128
                }
            }

            impl Sub<$t> for &BigInteger
            {
                type Output = BigInteger;

                fn sub(self, rhs: $t) -> Self::Output
                {
                    self - rhs as i128
                }
            }
        )*
    };
}

impl_bigint_sub_signed!(i8, i16, i32, i64, isize);

impl Neg for BigInteger
{
    type Output = Self;

    fn neg(self) -> Self::Output
    {
        BigInteger::create(self.mag, -self.signum)
    }
}

impl Neg for &BigInteger
{
    type Output = BigInteger;

    fn neg(self) -> Self::Output
    {
        BigInteger::create(self.mag.clone(), -self.signum)
    }
}

impl Shl<usize> for BigInteger
{
    type Output = Self;

    fn shl(self, rhs: usize) -> Self::Output
    {
        if self.signum == 0 || rhs == 0
        {
            Self::zero()
        }
        else
        {
            BigInteger::create(shift_slice_left(&self.mag, rhs).into(), self.signum)
        }
    }
}

impl Shl<usize> for &BigInteger
{
    type Output = BigInteger;

    fn shl(self, rhs: usize) -> Self::Output
    {
        if self.signum == 0
        {
            BigInteger::zero()
        }
        else
        {
            self.clone() << rhs
        }
    }
}

macro_rules! impl_bigint_shl_unsigned {
    ($($t:ty),*) => {
        $(
            impl Shl<$t> for BigInteger
            {
                type Output = Self;

                fn shl(self, rhs: $t) -> Self::Output
                {
                    self << rhs as usize
                }
            }

            impl Shl<$t> for &BigInteger
            {
                type Output = BigInteger;

                fn shl(self, rhs: $t) -> Self::Output
                {
                    self << rhs as usize
                }
            }
        )*
    };
}

impl_bigint_shl_unsigned!(u8, u16, u32, u64);

impl Shl<isize> for BigInteger
{
    type Output = Self;

    fn shl(self, rhs: isize) -> Self::Output
    {
        if self.signum == 0
        {
            Self::zero()
        }
        else if rhs == 0
        {
            self
        }
        else if rhs < 0
        {
            self >> rhs.unsigned_abs()
        }
        else
        {
            self << rhs.cast_unsigned()
        }
    }
}

impl Shl<isize> for &BigInteger
{
    type Output = BigInteger;

    fn shl(self, rhs: isize) -> Self::Output
    {
        if self.signum == 0
        {
            BigInteger::zero()
        }
        else if rhs == 0
        {
            self.clone()
        }
        else if rhs < 0
        {
            self >> rhs.unsigned_abs()
        }
        else
        {
            self << rhs.cast_unsigned()
        }
    }
}

macro_rules! impl_bigint_shl_signed {
    ($($t:ty),*) => {
        $(
            impl Shl<$t> for BigInteger
            {
                type Output = Self;

                fn shl(self, rhs: $t) -> Self::Output
                {
                    self << rhs as isize
                }
            }

            impl Shl<$t> for &BigInteger
            {
                type Output = BigInteger;

                fn shl(self, rhs: $t) -> Self::Output
                {
                    self << rhs as isize
                }
            }
        )*
    };
}

impl_bigint_shl_signed!(i8, i16, i32, i64);

impl Shr<usize> for BigInteger
{
    type Output = Self;

    fn shr(self, rhs: usize) -> Self::Output
    {
        todo!()
    }
}

impl Shr<usize> for &BigInteger
{
    type Output = BigInteger;

    fn shr(self, rhs: usize) -> Self::Output
    {
        if self.signum == 0
        {
            BigInteger::zero()
        }
        else
        {
            self.clone() >> rhs
        }
    }
}

macro_rules! impl_bigint_shr_unsigned {
    ($($t:ty),*) => {
        $(
            impl Shr<$t> for BigInteger
            {
                type Output = Self;

                fn shr(self, rhs: $t) -> Self::Output
                {
                    self >> rhs as usize
                }
            }

            impl Shr<$t> for &BigInteger
            {
                type Output = BigInteger;

                fn shr(self, rhs: $t) -> Self::Output
                {
                    self >> rhs as usize
                }
            }
        )*
    };
}

impl_bigint_shr_unsigned!(u8, u16, u32, u64);

impl Shr<isize> for BigInteger
{
    type Output = Self;

    fn shr(self, rhs: isize) -> Self::Output
    {
        todo!()
    }
}

impl Shr<isize> for &BigInteger
{
    type Output = BigInteger;

    fn shr(self, rhs: isize) -> Self::Output
    {
        todo!()
    }
}

macro_rules! impl_bigint_shr_signed {
    ($($t:ty),*) => {
        $(
            impl Shr<$t> for BigInteger
            {
                type Output = Self;

                fn shr(self, rhs: $t) -> Self::Output
                {
                    self >> rhs as isize
                }
            }

            impl Shr<$t> for &BigInteger
            {
                type Output = BigInteger;

                fn shr(self, rhs: $t) -> Self::Output
                {
                    self >> rhs as isize
                }
            }
        )*
    };
}

impl_bigint_shr_signed!(i8, i16, i32, i64);

impl BitAnd for BigInteger
{
    type Output = Self;

    fn bitand(self, rhs: Self) -> Self::Output
    {
        if self.signum == 0 || rhs.signum == 0
        {
            BigInteger::zero()
        }
        else
        {
            let sign = if self.signum == -1 && rhs.signum == -1 { -1 } else { 1 };

            let mut mag = vec![0; usize::max(self.mag.len(), rhs.mag.len())];

            for i in 0..mag.len()
            {
                mag[i] = self.get_int(mag.len() - i - 1)
                    &  rhs.get_int(mag.len() - i - 1)
            }

            todo!()
        }
    }
}

impl Not for BigInteger
{
    type Output = Self;

    fn not(self) -> Self::Output
    {
        let mut mag = self.mag.into_vec();

        for i in 0..mag.len()
        {
            mag[i] = !mag[i];
        }

        mag = trusted_strip_leading_zero_ints(mag);

        BigInteger::create(mag.into(), -self.signum)
    }
}

impl Not for &BigInteger
{
    type Output = BigInteger;

    fn not(self) -> Self::Output
    {
        !self.clone()
    }
}

impl PartialEq for BigInteger
{
    fn eq(&self, rhs: &Self) -> bool
    {
        self.signum == rhs.signum && self.mag == rhs.mag
    }
}

impl PartialEq<u128> for BigInteger
{
    fn eq(&self, rhs: &u128) -> bool
    {
        if self.signum < 0
        {
            return false;
        }

        if self.signum == 0 && *rhs == 0
        {
            return true;
        }

        let arr = u128_to_u32_array(*rhs);

        let first_nonzero = arr.iter().position(|&n| n != 0).unwrap_or(arr.len());

        cmp_slices(&self.mag, &arr[first_nonzero..]) == Ordering::Equal
    }
}

macro_rules! impl_bigint_partialeq_unsigned {
    ($($t:ty),*) => {
        $(
            impl PartialEq<$t> for BigInteger
            {
                fn eq(&self, rhs: &$t) -> bool
                {
                    self == &(*rhs as u128)
                }
            }
        )*
    };
}

impl_bigint_partialeq_unsigned!(u8, u16, u32, u64, usize);

impl PartialEq<i128> for BigInteger
{
    fn eq(&self, rhs: &i128) -> bool
    {
        if self.signum.cmp(&0) != rhs.cmp(&0)
        {
            return false;
        }

        if self.signum == 0 && *rhs == 0
        {
            return true;
        }

        let abs = rhs.unsigned_abs();

        let arr = u128_to_u32_array(abs);

        let first_nonzero = arr.iter().position(|&n| n != 0).unwrap_or(arr.len());

        cmp_slices(&self.mag, &arr[first_nonzero..]) == Ordering::Equal
    }
}

macro_rules! impl_bigint_partialeq_signed {
    ($($t:ty),*) => {
        $(
            impl PartialEq<$t> for BigInteger
            {
                fn eq(&self, rhs: &$t) -> bool
                {
                    self == &(*rhs as i128)
                }
            }
        )*
    };
}

impl_bigint_partialeq_signed!(i8, i16, i32, i64, isize);

impl Eq for BigInteger {}

impl PartialOrd for BigInteger
{
    fn partial_cmp(&self, other: &Self) -> Option<Ordering>
    {
        Some(self.cmp(other))
    }
}

impl Ord for BigInteger
{
    fn cmp(&self, other: &Self) -> Ordering
    {
        if self.signum == other.signum
        {
            cmp_slices(&self.mag, &other.mag)
        }
        else
        {
            self.signum.cmp(&other.signum)
        }
    }
}

impl Debug for BigInteger
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        let mut mag_str = "[".to_owned();
        self.mag
            .iter()
            .for_each(|n| mag_str.push_str(&format!("0x{:08x}, ", n)));
        if mag_str.ends_with(", ")
        {
            mag_str.pop();
            mag_str.pop();
        }
        mag_str.push(']');

        write!(f, "BigInteger(sign: {}, mag: {})", self.signum, mag_str)
    }
}

// TODO integrate signum
impl LowerHex for BigInteger
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        let mut res = String::new();
        if self.mag.len() == 0
        {
            res.push_str("0");
        }
        else
        {
            res.push_str(&format!("{:x}", self.mag[0]));
        }

        for i in 1..self.mag.len()
        {
            res.push_str(&format!("{:08x}", self.mag[i]));
        }

        write!(f, "{}", res)
    }
}

impl UpperHex for BigInteger
{
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result
    {
        f.write_str(&format!("{:x}", self).to_uppercase())
    }
}

#[derive(Clone)]
struct BigIntegerInfo
{
    bit_cnt:                OnceLock<u64>,
    mag_bit_len:            OnceLock<u64>,
    lowest_set_bit:         OnceLock<Option<u64>>,
    num_trailing_zero_ints: OnceLock<u64>,
}

impl Default for BigIntegerInfo
{
    fn default() -> Self
    {
        Self {
            bit_cnt:                OnceLock::new(),
            mag_bit_len:            OnceLock::new(),
            lowest_set_bit:         OnceLock::new(),
            num_trailing_zero_ints: OnceLock::new(),
        }
    }
}

const BITS_PER_DIGIT: [u16; 37] = [0, 0,
    1024, 1624, 2048, 2378, 2648, 2875, 3072, 3247, 3402, 3543,
    3672, 3790, 3899, 4001, 4096, 4186, 4271, 4350, 4426, 4498,
    4567, 4633, 4696, 4756, 4814, 4870, 4923, 4975, 5025, 5074,
    5120, 5166, 5210, 5253, 5295
];

const DIGITS_PER_INT: [u8; 37] = [0, 0,
    30, 19, 15, 13, 11, 11, 10, 9, 9, 8,
    8,  8,  8,  7,  7,  7,  7, 7, 7, 7,
    6,  6,  6,  6,  6,  6,  6, 6, 6, 6,
    6,  6,  6,  6,  5
];

const INT_RADIX: [u32; 35] = [
    0x40000000, 0x4546b3db, 0x40000000, 0x48c27395, 0x159fd800,
    0x75db9c97, 0x40000000, 0x17179149, 0x3b9aca00, 0x0cc6db61,
    0x19a10000, 0x309f1021, 0x57f6c100, 0xa2f1b6f,  0x10000000,
    0x18754571, 0x247dbc80, 0x3547667b, 0x4c4b4000, 0x6b5a6e1d,
    0x06c20a40, 0x08d2d931, 0x0b640000, 0x0e8d4a51, 0x1269ae40,
    0x17179149, 0x1cb91000, 0x23744899, 0x2b73a840, 0x34e63b41,
    0x40000000, 0x4cfa3cc1, 0x5c13d840, 0x6d91b519, 0x039aa400
];

/// Multiply x array times word y in place, and add word z
fn destructive_mul_add(x: &mut [u32], y: u32, z: u32)
{
    // Perform the multiplication word by word
    let y_long = y as u64;
    let z_long = z as u64;

    let len = x.len();

    let mut product: u64;
    let mut carry: u64 = 0;

    for i in (0..=len-1).rev()
    {
        product = y_long * (x[i] as u64) + carry;
        x[i] = product as u32;
        carry = product >> 32;
    }

    // Perform the addition
    let mut sum: u64 = (x[len - 1] as u64) + z_long;
    x[len - 1] = sum as u32;
    carry = sum >> 32;

    // println!("len={}", len);
    let max_bound = len.checked_sub(2).unwrap_or(0);

    for i in (0..=max_bound).rev()
    {
        println!("{}", i);
        sum = (x[i] as u64) + carry;
        x[i] = sum as u32;
        carry = sum >> 32;
    }
}

fn check_from_index_size(from: usize, size: usize, len: usize) -> Result<usize>
{
    if size <= len - from { Ok(from) } else { Err(anyhow!("Index out of bounds")) }
}

fn strip_leading_zero_ints(val: Vec<u32>, trusted: bool) -> Vec<u32>
{
    let mut keep: usize = 0;

    while keep < val.len() && val[keep] == 0 { keep += 1; }

    if trusted && keep == 0 {
        val
    } else {
        val.iter()
           .skip(keep)
           .copied()
           .collect()
    }
}

fn trusted_strip_leading_zero_ints(val: Vec<u32>) -> Vec<u32>
{
    strip_leading_zero_ints(val, true)
}

/// Returns a copy of the input array stripped of any leading zero bytes.
/// The returned array is either empty, or its 0-th element is non-zero,
/// meeting the "minimal" requirement for field `mag` (see comment on mag).
/// The range `[from, from + len)` must be well-formed w.r.t. array `arr`.
/// `b < -128` means that `arr[from]` has not yet been read.
/// Otherwise, `b` must be `arr[from]`, have been read only once before invoking
/// this method.
fn strip_leading_zero_bytes(mut b: u8, bytes: &[u8], mut from: usize, len: usize) -> Vec<u32>
{
    if len == 0
    {
        return Vec::new();
    }

    let to = from + len;

    from += 1;

    while b == 0 && from < to
    {
        b = bytes[from];
        from += 1;
    }

    if b == 0
    {
        return Vec::new();
    }

    let mut res = vec![0u32; ((to - from) >> 2) + 1];

    let mut d0 = b as u32;

    while ((to - from) & 0x3) != 0
    {
        d0 = d0 << 8 | bytes[from] as u32;
        from += 1;
    }
    res[0] = d0;

    let mut i = 1;

    while from < to
    {
        res[i] = (bytes[from]   as u32) << 24 |
            (bytes[from+1] as u32) << 16 |
            (bytes[from+2] as u32) << 8  |
            (bytes[from+3] as u32);

        from += 4;
        i    += 1;
    }

    res
}

/// Takes an array `arr` representing a negative 2's-complement number and
/// returns the minimal (no leading zero bytes) unsigned whose value is `-arr`.
///
/// The range `[from, from + len)` must be well-formed w.r.t. array `arr`.
/// `b` is assumed to be the result of reading `arr[from]` and to meet `b & 0x80 > 0`.
fn make_positive(mut b: u8, bytes: &[u8], mut from: usize, len: usize) -> Vec<u32>
{
    /*
     * By assumption, b & 0x80 > 0 (i.e. b is "negative").
     *
     * First collect the bytes into the resulting array res.
     * Then convert res to two's complement.
     *
     * Except for b, each read access to the input array a
     * is of the form a[from++].
     * The index from is never otherwise altered, except right below,
     * and only increases in steps of 1, always up to index to.
     * Hence, each byte in the array is read exactly once, from lower to
     * higher indices (from most to least significant byte).
     */
    let to = from + len;

    // b == a[from] has been read exactly once, skip to next index.
    from += 1;

    // Skip leading "-1" bytes.
    while b == u8::MAX && from < to
    {
        b = bytes[from];
        from += 1;
    }

    /*
     * A "digit" is a group of 4 adjacent bytes aligned w.r.t. index "to".
     * b is the most significant byte not -1, or -1 only if "from == to".
     *
     * Digit d0 spans the range of indices that includes current (from - 1).
     * (Implied -1 bytes are prepended to array arr as needed.)
     * It usually corresponds to res[0], except for the special case below.
     */
    let mut d0 = 0xffff_ff00 | b as u32;

    while ((to - from) & 0x03) != 0
    {
        b = bytes[from];
        from += 1;

        d0 = d0 << 8 | b as u32;
    }
    let f = from; // keeps the current "from" value for sizing purposes later

    // Skip zeros adjacent to d0, if at all.
    while b == 0 && from < to
    {
        b = bytes[from];
        from += 1;
    }

    /*
     * b is the most significant non-zero byte at or after (f - 1),
     * or 0 only if from == to.
     * Digit d spans the range of indices that includes (f - 1).
     */
    let mut d = b as u32;

    while ((to - from) & 0x03) != 0
    {
        d = d << 8 | bytes[from] as u32;

        from += 1;
    }

    /*
     * If the situation here is like this:
     * index:                      f                        to == from
     *      ..., -1,-1,  0,0,0,0,  0,0,0,0,  ...,  0,0,0,0
     * digit:               d0                        d
     * then, as shown, the number of zeros is a positive multiple of 4.
     * The array res needs a minimal length of (1 + 1 + (to - f) / 4)
     * to accommodate the two's complement, including a leading 1.
     * In any other case, there is at least one byte that is non-zero.
     * The array for the two's complement has length (0 + 1 + (to - f) / 4).
     * c is 1, resp., 0 for the two situations.
     */
    let c = if (to - from | d0 as usize | d as usize) == 0 { 1 } else { 0 };

    let res_len = c + 1 + ((to - f) >> 2);

    let mut res = vec![0; res_len];

    res[0] = if c == 0 { d0 } else { u32::MAX };

    let mut i = res.len() - ((to - from) >> 2);

    if i > 1 {
        res[i - 1] = d;
    }

    /*
     * Prepare the remaining digits.
     * (to - from) is a multiple of 4, so prepare an int for every 4 bytes.
     * This is a candidate for Unsafe.copy[Swap]Memory().
     */
    while from < to
    {
        res[i] = (bytes[from]   as u32) << 24 |
            (bytes[from+1] as u32) << 16 |
            (bytes[from+2] as u32) << 8  |
            (bytes[from+3] as u32);

        from += 4;
        i    += 1;
    }

    // Convert to two's complement. Here, i == res.length
    i -= 1;
    while i != usize::MAX && res[i] == 0
    {
        i -= 1;
    }

    res[i] = !res[i] + 1;

    i -= 1;
    while i != usize::MAX
    {
        res[i] = !res[i];
    }

    res
}

fn check_range(mag: &[u32]) -> Result<()>
{
    if (mag.len() > MAX_MAG_LENGTH) || (mag.len() == MAX_MAG_LENGTH && (mag[0] & 0x8000_0000) > 0) {
        Err(anyhow!("BigInteger would overflow supported range"))
    } else {
        Ok(())
    }
}

fn cmp_slices(x: &[u32], y: &[u32]) -> Ordering
{
    if x.len() < y.len()
    {
        Ordering::Less
    }
    else if x.len() > y.len()
    {
        Ordering::Greater
    }
    else
    {
        let mut cmp = Ordering::Equal;

        for i in 0..x.len()
        {
            cmp = x[i].cmp(&y[i]);
            if cmp != Ordering::Equal { break; }
        }

        cmp
    }
}

fn cmp_slice_to_u128(x: &[u32], y: u128) -> Ordering
{
    let arr = u128_to_u32_array(y);

    let first_nonzero = x.iter().position(|&n| n != 0).unwrap_or(x.len());

    cmp_slices(x, &arr[first_nonzero..])
}

fn cmp_slice_to_long(x: &[u32], y: u64) -> Ordering
{
    if x.len() > 2
    {
        return Ordering::Greater;
    }

    let high = (y >> 32) as u32;

    if high == 0
    {
        if x.len() < 1
        {
            Ordering::Less
        }
        else if x.len() > 1
        {
            Ordering::Greater
        }
        else
        {
            let a = x[0];
            let b = y as u32;
            if a != b { a.cmp(&b) } else { Ordering::Equal }
        }
    }
    else
    {
        if x.len() < 2 { return Ordering::Less; }

        let a = x[0];
        let b = high;

        if a != b { return a.cmp(&b); }

        let a = x[1];
        let b = y as u32;

        if a != b { a.cmp(&b) } else { Ordering::Equal }
    }
}

/// Returns `1` if `ord as i32 == signum`; `-1` otherwise.
///
/// Alternatively, this function can be interpreted as returning
/// `1` if `ord == signum.cmp(&0)` and `-1` otherwise.
fn get_signum_for_ord(ord: Ordering, signum: i32) -> i32
{
    if ord as i32 == signum { 1 } else { -1 }
}

fn add_slices(x: &[u32], y: &[u32]) -> Vec<u32>
{
    if x.len() == 0 { return y.to_vec(); }
    if y.len() == 0 { return x.to_vec(); }

    if x.len() < y.len()
    {
        return add_slices(y, x);
    }

    let mut x_idx = x.len();
    let mut y_idx = y.len();

    let mut res = vec![0u32; x_idx];

    let mut sum: u64 = 0;

    if y_idx == 1
    {
        x_idx -= 1;
        y_idx -= 1;

        sum = x[x_idx] as u64 + y[y_idx] as u64 + (sum >> 32);

        res[x_idx] = sum as u32;
    }
    else
    {
        let mut y_ovf = false;

        while !y_ovf
        {
            x_idx -= 1;
            (y_idx, y_ovf) = y_idx.overflowing_sub(1);

            sum = x[x_idx] as u64 + y[y_idx] as u64 + (sum >> 32);

            res[x_idx] = sum as u32;
        }
    }

    let mut carry = (sum >> 32) != 0;

    let mut x_ovf = false;
    while !x_ovf && carry
    {
        (x_idx, x_ovf) = x_idx.overflowing_sub(1);

        (res[x_idx], carry) = x[x_idx].overflowing_add(1);
    }

    while !x_ovf
    {
        (x_idx, x_ovf) = x_idx.overflowing_sub(1);

        res[x_idx] = x[x_idx];
    }

    if carry
    {
        res.insert(0, 0x01);
    }

    res
}

fn add_u128_to_slice(x: &[u32], y: u128) -> Vec<u32>
{
    let rhs = u128_to_u32_array(y);

    let first_nonzero = x.iter().position(|&x| x != 0).unwrap_or(x.len());

    add_slices(x, &rhs[first_nonzero..])
}

fn subtract_slices(big: &[u32], little: &[u32]) -> Vec<u32>
{
    assert!(big.len() >= little.len());

    let mut big_idx = big.len();
    let mut little_idx = little.len();

    let mut res = vec![0u32; big_idx];

    let mut diff: i64 = 0;

    let mut little_ovf = false;
    while !little_ovf
    {
        big_idx -= 1;
        (little_idx, little_ovf) = little_idx.overflowing_sub(1);

        diff = big[big_idx] as i64 - little[little_idx] as i64 + (diff >> 32);

        res[big_idx] = diff as u32;
    }

    let mut borrow = (diff >> 32) != 0;

    let mut big_ovf = false;
    while !big_ovf && borrow
    {
        (big_idx, big_ovf) = big_idx.overflowing_sub(1);

        (res[big_idx], borrow) = big[big_idx].overflowing_sub(1);
    }

    while !big_ovf
    {
        big_idx -= 1;
        res[big_idx] = big[big_idx];
    }

    res
}

fn subtract_u128_from_slice(big: &[u32], little: u128) -> Vec<u32>
{
    let rhs = u128_to_u32_array(little);

    let first_nonzero = big.iter().position(|&x| x != 0).unwrap_or(big.len());

    subtract_slices(big, &rhs[first_nonzero..])
}

fn subtract_slice_from_u128(big: u128, little: &[u32]) -> Vec<u32>
{
    let lhs = u128_to_u32_array(big);

    let first_nonzero = little.iter().position(|&x| x != 0).unwrap_or(little.len());

    subtract_slices(&lhs[first_nonzero..], little)
}

fn shift_slice_left(mag: &[u32], shift: usize) -> Vec<u32>
{
    let num_ints = shift >> 5; // Divide by 32
    let num_bits = (shift & 0x1f) as u32; // Modulo 32

    let mut res: Vec<u32>;
    if num_bits == 0
    {
        // No bit-shift, just copy the input array with "num_ints" number of zeros appended to the end.
        res = mag.to_vec();
        res.resize(res.len() + num_ints, 0);
    }
    else
    {
        let mut i = 0;
        let num_bits_inv = 32 - num_bits;
        let high_bits = mag[0] >> num_bits_inv;

        if high_bits != 0
        {
            res = vec![0; mag.len() + num_ints + 1];
            res[i] = high_bits;
            i += 1;
        }
        else
        {
            res = vec![0; mag.len() + num_ints];
        }

        let num_iter = mag.len() - 1;
        shift_left_worker(&mut res, mag, num_bits, i, num_iter);
        res[num_iter + i] = mag[num_iter] << num_bits;
    }
    res
}

fn shift_left_worker(new: &mut [u32], old: &[u32], shift: u32, mut new_idx: usize, len: usize)
{
    let shift_right = 32 - shift;
    let mut old_idx = 0;
    while old_idx < len
    {
        new[new_idx] = (old[old_idx] << shift) | (old[old_idx + 1] >> shift_right);
        new_idx += 1;
        old_idx += 1;
    }
}

const fn u128_to_u32_array(x: u128) -> [u32; 4]
{
    [
        (x >> 96) as u32,
        (x >> 64) as u32,
        (x >> 32) as u32,
        x as u32,
    ]
}

#[cfg(test)]
mod tests
{
    use super::*;

    #[test]
    fn test_from_unsigned()
    {
        let mut bigint = BigInteger::from(0x01020304_05060708_090a0b0c_0d0e0f10u128);

        assert_eq!(&*bigint.mag, &[0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10], "BigInteger::from(u128).mag");
        assert_eq!(bigint.signum(), 1, "BigInteger::from(u128).sign");

        bigint = BigInteger::from(0x01020304_05060708u64);

        assert_eq!(&*bigint.mag, &[0x01020304, 0x05060708], "BigInteger::from(u64).mag");
        assert_eq!(bigint.signum(), 1, "BigInteger::from(u64).sign");

        bigint = BigInteger::from(0x01020304u32);

        assert_eq!(&*bigint.mag, &[0x01020304], "BigInteger::from(u32).mag");
        assert_eq!(bigint.signum(), 1, "BigInteger::from(u32).sign");

        bigint = BigInteger::from(0x0102u16);

        assert_eq!(&*bigint.mag, &[0x0102], "BigInteger::from(u16).mag");
        assert_eq!(bigint.signum(), 1, "BigInteger::from(u16).sign");

        bigint = BigInteger::from(0x01u8);

        assert_eq!(&*bigint.mag, &[0x01], "BigInteger::from(u8).mag");
        assert_eq!(bigint.signum(), 1, "BigInteger::from(u8).sign");
    }

    #[test]
    fn test_from_signed()
    {
        let mut bigint = BigInteger::from(-0x01020304_05060708_090a0b0c_0d0e0f10i128);

        assert_eq!(&*bigint.mag, &[0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10], "BigInteger::from(i128).mag");
        assert_eq!(bigint.signum(), -1, "BigInteger::from(i128).sign");

        bigint = BigInteger::from(-0x01020304_05060708i64);

        assert_eq!(&*bigint.mag, &[0x01020304, 0x05060708], "BigInteger::from(i64)");
        assert_eq!(bigint.signum(), -1, "BigInteger::from(i64)");

        bigint = BigInteger::from(-0x01020304i32);

        assert_eq!(&*bigint.mag, &[0x01020304], "BigInteger::from(i32)");
        assert_eq!(bigint.signum(), -1, "BigInteger::from(i32)");

        bigint = BigInteger::from(-0x0102i16);

        assert_eq!(&*bigint.mag, &[0x0102], "BigInteger::from(i16)");
        assert_eq!(bigint.signum(), -1, "BigInteger::from(i16)");

        bigint = BigInteger::from(-0x01i8);

        assert_eq!(&*bigint.mag, &[0x01], "BigInteger::from(i8)");
        assert_eq!(bigint.signum(), -1, "BigInteger::from(i8)");
    }

    #[test]
    fn test_from_pos_neg()
    {
        let mut bigint = BigInteger::from(0x01020304i32);

        assert_eq!(&*bigint.mag, &[0x01020304], "BigInteger::from(i32)");
        assert_eq!(bigint.signum(), 1, "BigInteger::from(i32)");

        bigint = BigInteger::from(-0x01020304i32);

        assert_eq!(&*bigint.mag, &[0x01020304], "BigInteger::from(i32)");
        assert_eq!(bigint.signum(), -1, "BigInteger::from(i32)");
    }

    #[test]
    fn test_from_bytes()
    {
        let bytes: &'static [u8] = &[0x01, 0x02, 0x03, 0x04, 0x05, 0x06, 0x07, 0x08, 0x09, 0x0a, 0x0b, 0x0c, 0x0d, 0x0e, 0x0f, 0x10];
        let bigint = BigInteger::try_from(bytes).unwrap();

        assert_eq!(&*bigint.mag, &[0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10], "BigInteger::try_from(bytes).mag");
        assert_eq!(bigint.signum(), 1, "BigInteger::try_from(bytes).sign");

        let bytes2: &'static [u8] = &[0xfe, 0xfd, 0xfc, 0xfb, 0xfa, 0xf9, 0xf8, 0xf7, 0xf6, 0xf5, 0xf4, 0xf3, 0xf2, 0xf1, 0xf0, 0xf0];
        let bigint2 = BigInteger::try_from(bytes2).unwrap();

        assert_eq!(&*bigint2.mag, &[0x01020304, 0x05060708, 0x090a0b0c, 0x0d0e0f10], "BigInteger::try_from(bytes).mag");
        assert_eq!(bigint2.signum(), -1, "BigInteger::try_from(bytes).sign");
    }

    #[test]
    fn test_from_empty_bytes()
    {
        let bytes: &'static [u8] = &[];
        assert!(BigInteger::try_from(bytes).is_err());
    }
}
