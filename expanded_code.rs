#![feature(prelude_import)]
#[prelude_import]
use std::prelude::rust_2021::*;
#[macro_use]
extern crate std;
mod shape {
    use std::error::Error;
    use std::fmt::Debug;
    use std::boxed::Box;
    pub struct Shape(Vec<usize>);
    #[automatically_derived]
    impl ::core::clone::Clone for Shape {
        #[inline]
        fn clone(&self) -> Shape {
            Shape(::core::clone::Clone::clone(&self.0))
        }
    }
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for Shape {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for Shape {
        #[inline]
        fn eq(&self, other: &Shape) -> bool {
            self.0 == other.0
        }
    }
    #[automatically_derived]
    impl ::core::cmp::Eq for Shape {
        #[inline]
        #[doc(hidden)]
        #[coverage(off)]
        fn assert_receiver_is_total_eq(&self) -> () {
            let _: ::core::cmp::AssertParamIsEq<Vec<usize>>;
        }
    }
    pub const SCALAR: Shape = Shape(::alloc::vec::Vec::new());
    impl Debug for Shape {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            f.write_fmt(format_args!("Shape({0:?})", self.0))
        }
    }
    impl<const C: usize> From<[usize; C]> for Shape {
        fn from(shape: [usize; C]) -> Self {
            Self(shape.to_vec())
        }
    }
    impl<const C: usize> From<&[usize; C]> for Shape {
        fn from(shape: &[usize; C]) -> Self {
            Self(shape.to_vec())
        }
    }
    impl From<Vec<usize>> for Shape {
        fn from(shape: Vec<usize>) -> Self {
            Self(shape)
        }
    }
    impl From<&[usize]> for Shape {
        fn from(shape: &[usize]) -> Self {
            Self(shape.to_vec())
        }
    }
    impl From<(usize, usize)> for Shape {
        fn from(shape: (usize, usize)) -> Self {
            Self(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([shape.0, shape.1]),
                ),
            )
        }
    }
    impl From<(usize, usize, usize)> for Shape {
        fn from(shape: (usize, usize, usize)) -> Self {
            Self(
                <[_]>::into_vec(
                    #[rustc_box]
                    ::alloc::boxed::Box::new([shape.0, shape.1, shape.2]),
                ),
            )
        }
    }
    impl From<usize> for Shape {
        fn from(shape: usize) -> Self {
            Self(<[_]>::into_vec(#[rustc_box] ::alloc::boxed::Box::new([shape])))
        }
    }
    impl From<()> for Shape {
        fn from(_: ()) -> Self {
            Self(::alloc::vec::Vec::new())
        }
    }
    impl Shape {
        pub fn from_dims(dims: &[usize]) -> Self {
            Self(dims.to_vec())
        }
        pub fn into_dims(self) -> Vec<usize> {
            self.0
        }
        pub fn rank(&self) -> usize {
            self.0.len()
        }
        pub fn dims(&self) -> &[usize] {
            &self.0
        }
        pub fn new(shape: Vec<usize>) -> Self {
            Self(shape)
        }
        pub fn len(&self) -> usize {
            self.0.len()
        }
        pub fn size(&self) -> usize {
            self.0.iter().fold(1, |acc, &x| acc * x)
        }
        pub fn get(&self, index: usize) -> Option<&usize> {
            self.0.get(index)
        }
        pub fn as_slice(&self) -> &[usize] {
            &self.0
        }
        pub fn elem_count(&self) -> usize {
            self.0.iter().product()
        }
        pub fn stride_contiguous(&self) -> Vec<usize> {
            let mut stride: Vec<_> = self
                .0
                .iter()
                .rev()
                .scan(
                    1,
                    |prod, u| {
                        let prod_pre_mult = *prod;
                        *prod *= u;
                        Some(prod_pre_mult)
                    },
                )
                .collect();
            stride.reverse();
            stride
        }
        pub fn is_contiguous(&self, stride: &[usize]) -> bool {
            if self.0.len() != stride.len() {
                return false;
            }
            let mut acc = 1;
            for (s, &x) in stride.iter().zip(self.0.iter()).rev() {
                if s != &acc {
                    return false;
                }
                acc *= x;
            }
            true
        }
        pub fn extend(&self, dims: &[usize]) -> Shape {
            let mut new_dims = self.0.clone();
            new_dims.extend_from_slice(dims);
            Shape(new_dims)
        }
        pub fn is_f_contiguous(&self, stride: &[usize]) -> bool {
            if self.0.len() != stride.len() {
                return false;
            }
            let mut acc = 1;
            for (s, &x) in stride.iter().zip(self.0.iter()) {
                if s != &acc {
                    return false;
                }
                acc *= x;
            }
            true
        }
        pub fn broadcast_shape_binary_to(
            &self,
            rhs: &Self,
            op: &'static str,
        ) -> Result<Shape, Box<dyn Error>> {
            let lhs = self;
            let lhs_dims = lhs.dims();
            let rhs_dims = rhs.dims();
            let lhs_ndims = lhs_dims.len();
            let rhs_ndims = rhs_dims.len();
            let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
            let mut bcast_dims = ::alloc::vec::from_elem(0, bcast_ndims);
            for (idx, bcast_dim) in bcast_dims.iter_mut().enumerate() {
                let lhs_dim = if idx < lhs_ndims {
                    lhs_dims[lhs_ndims - 1 - idx]
                } else {
                    1
                };
                let rhs_dim = if idx < rhs_ndims {
                    rhs_dims[rhs_ndims - 1 - idx]
                } else {
                    1
                };
                if lhs_dim == 1 {
                    *bcast_dim = rhs_dim;
                } else if rhs_dim == 1 {
                    *bcast_dim = lhs_dim;
                } else if lhs_dim == rhs_dim {
                    *bcast_dim = lhs_dim;
                } else {
                    Err({
                        let res = ::alloc::fmt::format(
                            format_args!(
                                "Incompatible shapes for broadcasting: {0:?} {1:?} {2:?}",
                                lhs, op, rhs
                            ),
                        );
                        res
                    })?;
                }
            }
            bcast_dims.reverse();
            Ok(Shape::from_dims(&bcast_dims))
        }
        pub fn broadcast_shape_matmul(
            &self,
            rhs: &Self,
        ) -> Result<(Shape, Shape), Box<dyn Error>> {
            let lhs = self;
            let lhs_dims = lhs.dims();
            let rhs_dims = rhs.dims();
            let (m, lhs_k) = (
                lhs_dims[lhs_dims.len() - 2],
                lhs_dims[lhs_dims.len() - 1],
            );
            let (rhs_k, n) = (
                rhs_dims[rhs_dims.len() - 2],
                rhs_dims[rhs_dims.len() - 1],
            );
            if lhs_k != rhs_k {
                Err({
                    let res = ::alloc::fmt::format(
                        format_args!(
                            "Incompatible shapes for matrix multiplication: {0:?} {1:?}",
                            lhs, rhs
                        ),
                    );
                    res
                })?;
            }
            let lhs_b = Self::from(&lhs_dims[..lhs_dims.len() - 2]);
            let rhs_b = Self::from(&rhs_dims[..rhs_dims.len() - 2]);
            let bcast_shape = lhs_b
                .broadcast_shape_binary_to(&rhs_b, "broadcast_matmul")?;
            let bcast_dims = bcast_shape.dims();
            let bcast_lhs = [bcast_dims, &[m, lhs_k]].concat();
            let bcast_rhs = [bcast_dims, &[rhs_k, n]].concat();
            Ok((Shape::from(bcast_lhs), Shape::from(bcast_rhs)))
        }
    }
    pub trait ShapeWithOneHole {
        fn into_shape(self, el_count: usize) -> Result<Shape, Box<dyn Error>>;
    }
    impl<S: Into<Shape>> ShapeWithOneHole for S {
        fn into_shape(self, _el_count: usize) -> Result<Shape, Box<dyn Error>> {
            Ok(self.into())
        }
    }
}
mod tensor {
    use std::ops::Deref;
    use std::sync::RwLock;
    use std::sync::Arc;
    use crate::storage::Storage;
    use crate::layout::Layout;
    use crate::dtype::Dtype;
    use crate::device::{Device, NdArray};
    use crate::shape::{Shape, ShapeWithOneHole};
    use crate::op::*;
    use crate::cpu_backend::CpuStorage;
    pub struct TensorId(usize);
    #[automatically_derived]
    impl ::core::fmt::Debug for TensorId {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_tuple_field1_finish(f, "TensorId", &&self.0)
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for TensorId {
        #[inline]
        fn clone(&self) -> TensorId {
            let _: ::core::clone::AssertParamIsClone<usize>;
            *self
        }
    }
    #[automatically_derived]
    impl ::core::marker::Copy for TensorId {}
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for TensorId {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for TensorId {
        #[inline]
        fn eq(&self, other: &TensorId) -> bool {
            self.0 == other.0
        }
    }
    #[automatically_derived]
    impl ::core::cmp::Eq for TensorId {
        #[inline]
        #[doc(hidden)]
        #[coverage(off)]
        fn assert_receiver_is_total_eq(&self) -> () {
            let _: ::core::cmp::AssertParamIsEq<usize>;
        }
    }
    #[automatically_derived]
    impl ::core::cmp::PartialOrd for TensorId {
        #[inline]
        fn partial_cmp(
            &self,
            other: &TensorId,
        ) -> ::core::option::Option<::core::cmp::Ordering> {
            ::core::cmp::PartialOrd::partial_cmp(&self.0, &other.0)
        }
    }
    #[automatically_derived]
    impl ::core::hash::Hash for TensorId {
        #[inline]
        fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) -> () {
            ::core::hash::Hash::hash(&self.0, state)
        }
    }
    impl TensorId {
        fn new() -> Self {
            use std::sync::atomic;
            static COUNTER: atomic::AtomicUsize = atomic::AtomicUsize::new(1);
            Self(COUNTER.fetch_add(1, atomic::Ordering::Relaxed))
        }
    }
    pub struct Tensor_ {
        id: TensorId,
        storage: Arc<RwLock<Storage>>,
        layout: Layout,
        dtype: Dtype,
        device: Device,
        is_var: bool,
        op: BackpropOp,
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for Tensor_ {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            let names: &'static _ = &[
                "id",
                "storage",
                "layout",
                "dtype",
                "device",
                "is_var",
                "op",
            ];
            let values: &[&dyn ::core::fmt::Debug] = &[
                &self.id,
                &self.storage,
                &self.layout,
                &self.dtype,
                &self.device,
                &self.is_var,
                &&self.op,
            ];
            ::core::fmt::Formatter::debug_struct_fields_finish(
                f,
                "Tensor_",
                names,
                values,
            )
        }
    }
    pub struct Tensor(Arc<Tensor_>);
    #[automatically_derived]
    impl ::core::clone::Clone for Tensor {
        #[inline]
        fn clone(&self) -> Tensor {
            Tensor(::core::clone::Clone::clone(&self.0))
        }
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for Tensor {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_tuple_field1_finish(f, "Tensor", &&self.0)
        }
    }
    impl Deref for Tensor {
        type Target = Tensor_;
        fn deref(&self) -> &Self::Target {
            &self.0.as_ref()
        }
    }
    impl std::fmt::Display for Tensor {
        fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_fmt(
                format_args!(
                    "Tensor(id: {0:?}, shape: {1:?}, dtype: {2:?})", self.id, self.layout
                    .shape(), self.dtype
                ),
            )
        }
    }
    impl Tensor {
        pub fn new<A: NdArray>(
            array: A,
            device: &Device,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            Self::new_impl(array, device, false)
        }
        pub fn new_impl<A: NdArray>(
            array: A,
            device: &Device,
            is_var: bool,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = array.shape()?;
            let storage = device.storage(array)?;
            let op = BackpropOp::none();
            Ok(from_storage(storage, shape, op, is_var))
        }
        pub fn track_op(&self) -> bool {
            return self.is_var;
        }
        pub fn op(&self) -> &Option<Op> {
            &self.op
        }
        pub fn id(&self) -> TensorId {
            self.id
        }
        pub fn shape(&self) -> &Shape {
            &self.layout.shape()
        }
        pub fn layout(&self) -> &Layout {
            &self.layout
        }
        pub fn storage(&self) -> &Arc<RwLock<Storage>> {
            &self.storage
        }
        pub fn start_offset(&self) -> usize {
            self.layout.start_offset()
        }
        pub fn is_var(&self) -> bool {
            self.is_var
        }
        pub fn ones_impl<S: Into<Shape>>(
            shape: S,
            dtype: Dtype,
            device: &Device,
            is_val: bool,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let none = BackpropOp::none();
            let shape = shape.into();
            let storage = device.ones(&shape, &dtype)?;
            let tensor = from_storage(storage, shape, none, is_val);
            Ok(tensor)
        }
        pub fn rank(&self) -> usize {
            self.layout.shape().rank()
        }
        pub fn ones<S: Into<Shape>>(
            shape: S,
            dtype: Dtype,
            device: &Device,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            Self::ones_impl(shape, dtype, device, false)
        }
        pub fn zeros_impl<S: Into<Shape>>(
            shape: S,
            dtype: Dtype,
            device: &Device,
            is_val: bool,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let none = BackpropOp::none();
            let shape = shape.into();
            let storage = device.zeros(&shape, &dtype)?;
            let tensor = from_storage(storage, shape, none, is_val);
            Ok(tensor)
        }
        pub fn zeros<S: Into<Shape>>(
            shape: S,
            dtype: Dtype,
            device: &Device,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            Self::zeros_impl(shape, dtype, device, false)
        }
        pub fn rand_uniform<S: Into<Shape>>(
            lo: f64,
            hi: f64,
            shape: S,
            device: &Device,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = shape.into();
            let dtype = Dtype::F64;
            let storage = device.rand_uniform(lo, hi, &shape, &dtype)?;
            let none = BackpropOp::none();
            let tensor = from_storage(storage, shape, none, false);
            Ok(tensor)
        }
        pub fn rand_uniform_impl<S: Into<Shape>>(
            lo: f64,
            hi: f64,
            shape: S,
            device: &Device,
            is_val: bool,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = shape.into();
            let dtype = Dtype::F64;
            let storage = device.rand_uniform(lo, hi, &shape, &dtype)?;
            let none = BackpropOp::none();
            let tensor = from_storage(storage, shape, none, is_val);
            Ok(tensor)
        }
        pub fn randn<S: Into<Shape>>(
            mean: f64,
            std: f64,
            shape: S,
            device: &Device,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = shape.into();
            let dtype = Dtype::F64;
            let storage = device.randn(mean, std, &shape, &dtype)?;
            let none = BackpropOp::none();
            let tensor = from_storage(storage, shape, none, false);
            Ok(tensor)
        }
        pub fn randn_impl<S: Into<Shape>>(
            mean: f64,
            std: f64,
            shape: S,
            device: &Device,
            is_val: bool,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = shape.into();
            let dtype = Dtype::F64;
            let storage = device.randn(mean, std, &shape, &dtype)?;
            let none = BackpropOp::none();
            let tensor = from_storage(storage, shape, none, is_val);
            Ok(tensor)
        }
        pub fn t(&self) -> Result<Self, Box<dyn std::error::Error>> {
            let rank = self.rank();
            if rank < 2 {
                return Err("Invalid shape for transpose".into());
            }
            let dim1 = rank - 1;
            let dim2 = rank - 2;
            self.transpose(dim2, dim1)
        }
        pub fn transpose(
            &self,
            dim1: usize,
            dim2: usize,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            if dim1 == dim2 {
                return Ok(self.clone());
            }
            let layout = self.layout.transpose(dim1, dim2)?;
            let storage = self.storage.clone();
            let op = BackpropOp::none();
            let tensor = Tensor_ {
                id: TensorId::new(),
                storage,
                layout,
                device: self.device,
                dtype: self.dtype,
                is_var: self.is_var,
                op,
            };
            Ok(Tensor(Arc::new(tensor)))
        }
        pub fn make_var(&self) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = self.layout.shape();
            let mut storage = self.device.zeros(&shape, &self.dtype)?;
            let op = BackpropOp::none();
            self.storage
                .read()
                .unwrap()
                .copy_strided_src(&mut storage, 0, &self.layout)?;
            Ok(from_storage(storage, shape.clone(), op, true))
        }
        pub fn matmul(&self, other: &Self) -> Result<Self, Box<dyn std::error::Error>> {
            let (self_shape, other_shape) = (self.layout.shape(), other.layout.shape());
            let (self_dims, other_dims) = (self_shape.dims(), other_shape.dims());
            if self_dims.len() < 2 || other_dims.len() < 2 {
                return Err("Invalid shape for matmul".into());
            }
            let (self_m, self_k) = (
                self_dims[self_dims.len() - 2],
                self_dims[self_dims.len() - 1],
            );
            let (other_k, other_n) = (
                other_dims[other_dims.len() - 2],
                other_dims[other_dims.len() - 1],
            );
            let self_b = self_dims[..self_dims.len() - 2].iter().product();
            let other_b = other_dims[..other_dims.len() - 2].iter().product();
            if self_k != other_k || self_b != other_b {
                return Err("Invalid shape for matmul".into());
            }
            let c_shape = Shape::from(&self_dims[..self_dims.len() - 2])
                .extend(&[self_m, other_n]);
            let storage = self
                .storage
                .read()
                .unwrap()
                .matmul(
                    &*other.storage.read().unwrap(),
                    (self_b, self_m, other_n, self_k),
                    &self.layout,
                    &other.layout,
                )?;
            let op = BackpropOp::none();
            Ok(from_storage(storage, c_shape, op, false))
        }
        pub fn ones_like(&self) -> Result<Self, Box<dyn std::error::Error>> {
            return Tensor::ones(
                Shape::from(self.layout.dims()),
                self.dtype,
                &self.device,
            );
        }
        pub fn arange(
            start: f64,
            end: f64,
            step: f64,
            dtype: Dtype,
            device: &Device,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let mut data = ::alloc::vec::Vec::new();
            if step > 0.0 {
                let mut i = start;
                while i < end {
                    data.push(i);
                    i += step;
                }
            } else {
                let mut i = start;
                while i > end {
                    data.push(i);
                    i += step;
                }
            }
            let shape = Shape::from(data.len());
            let op = BackpropOp::none();
            let storage = device.storage_owned(data)?;
            Ok(from_storage(storage, shape, op, false))
        }
        pub fn contiguous(&self) -> Result<Self, Box<dyn std::error::Error>> {
            if self.layout.is_contiguous() {
                return Ok(self.clone());
            }
            let shape = self.layout.shape();
            let mut storage = self.device.zeros(&shape, &self.dtype)?;
            self.storage
                .read()
                .unwrap()
                .copy_strided_src(&mut storage, 0, &self.layout)?;
            let op = BackpropOp::none();
            Ok(from_storage(storage, shape.clone(), op, false))
        }
        pub fn reshape<S: ShapeWithOneHole>(
            &self,
            s: S,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = s.into_shape(self.layout.shape().elem_count())?;
            if shape.elem_count() != self.layout.shape().elem_count() {
                return Err("Invalid shape for reshape".into());
            }
            let op = BackpropOp::none();
            if self.layout.is_contiguous() {
                let tenosr_ = Tensor_ {
                    id: TensorId::new(),
                    storage: self.storage.clone(),
                    layout: Layout::contiguous_with_offset(
                        shape,
                        self.layout.start_offset(),
                    ),
                    device: self.device.clone(),
                    dtype: self.dtype,
                    is_var: self.is_var,
                    op,
                };
                Ok(Tensor(Arc::new(tenosr_)))
            } else {
                let mut storage = self.device.zeros(&shape, &self.dtype)?;
                self.storage
                    .read()
                    .unwrap()
                    .copy_strided_src(&mut storage, 0, &self.layout)?;
                Ok(from_storage(storage, shape, op, false))
            }
        }
        pub fn affine(
            &self,
            mul: f64,
            add: f64,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let storage = self.storage.read().unwrap().affine(&self.layout, mul, add)?;
            let op = BackpropOp::none();
            Ok(from_storage(storage, self.layout.shape().clone(), op, false))
        }
        pub fn narrow(
            &self,
            dim: usize,
            start: usize,
            len: usize,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let dims = self.layout.shape().dims();
            if dim >= dims.len() {
                return Err("Invalid dim for narrow".into());
            }
            if start + len > dims[dim] {
                return Err("Invalid start and len for narrow".into());
            }
            if start == 0 && len == dims[dim] {
                return Ok(self.clone());
            } else {
                let mut dims = dims.to_vec();
                dims[dim] = len;
                let layout = self.layout.narrow(dim, start, len)?;
                let tensor_ = Tensor_ {
                    id: TensorId::new(),
                    storage: self.storage.clone(),
                    layout,
                    device: self.device.clone(),
                    dtype: self.dtype,
                    is_var: self.is_var,
                    op: BackpropOp::none(),
                };
                Ok(Tensor(Arc::new(tensor_)))
            }
        }
        pub fn recip(&self) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = self.layout.shape();
            let storage = self
                .storage
                .read()
                .unwrap()
                .unary_op::<Reciprocal>(&self.layout)?;
            let op = BackpropOp::new1(self, |x| Op::Unary(x, UnaryOp::Reciprocal));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
        pub fn abs(&self) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = self.layout.shape();
            let storage = self.storage.read().unwrap().unary_op::<Abs>(&self.layout)?;
            let op = BackpropOp::new1(self, |x| Op::Unary(x, UnaryOp::Abs));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
        pub fn neg(&self) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = self.layout.shape();
            let storage = self.storage.read().unwrap().unary_op::<Neg>(&self.layout)?;
            let op = BackpropOp::new1(self, |x| Op::Unary(x, UnaryOp::Neg));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
        pub fn exp(&self) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = self.layout.shape();
            let storage = self.storage.read().unwrap().unary_op::<Exp>(&self.layout)?;
            let op = BackpropOp::new1(self, |x| Op::Unary(x, UnaryOp::Exp));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
        pub fn relu(&self) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = self.layout.shape();
            let storage = self.storage.read().unwrap().unary_op::<Relu>(&self.layout)?;
            let op = BackpropOp::new1(self, |x| Op::Unary(x, UnaryOp::Relu));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
        pub fn log(&self) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = self.layout.shape();
            let storage = self.storage.read().unwrap().unary_op::<Log>(&self.layout)?;
            let op = BackpropOp::new1(self, |x| Op::Unary(x, UnaryOp::Log));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
        pub fn sqr(&self) -> Result<Self, Box<dyn std::error::Error>> {
            let shape = self.layout.shape();
            let storage = self.storage.read().unwrap().unary_op::<Sqr>(&self.layout)?;
            let op = BackpropOp::new1(self, |x| Op::Unary(x, UnaryOp::Sqr));
            Ok(from_storage(storage, shape.clone(), op, false))
        }
        pub fn add(&self, other: &Self) -> Result<Self, Box<dyn std::error::Error>> {
            if self.layout.shape() != other.layout.shape() {
                return Err("Invalid shape for binary op".into());
            }
            let storage = self
                .storage
                .read()
                .unwrap()
                .binary_op::<
                    Add,
                >(&*other.storage.read().unwrap(), &self.layout, &other.layout)?;
            let op = BackpropOp::new1(
                self,
                |x| Op::Binary(x, other.clone(), BinaryOp::Add),
            );
            Ok(from_storage(storage, shape, op, false))
        }
    }
    impl std::ops::Add<Tensor> for f64 {
        type Output = Result<Tensor, Box<dyn std::error::Error>>;
        fn add(self, rhs: Tensor) -> Self::Output {
            rhs + self
        }
    }
    impl std::ops::Add<f64> for Tensor {
        type Output = Result<Tensor, Box<dyn std::error::Error>>;
        fn add(self, rhs: f64) -> Self::Output {
            let storage = self.storage.read().unwrap().affine(&self.layout, 1.0, rhs)?;
            let op = BackpropOp::none();
            Ok(from_storage(storage, self.layout.shape().clone(), op, false))
        }
    }
    impl std::ops::Sub<Tensor> for f64 {
        type Output = Result<Tensor, Box<dyn std::error::Error>>;
        fn sub(self, rhs: Tensor) -> Self::Output {
            rhs.affine(-1.0, self)
        }
    }
    impl std::ops::Sub<f64> for Tensor {
        type Output = Result<Tensor, Box<dyn std::error::Error>>;
        fn sub(self, rhs: f64) -> Self::Output {
            let storage = self.storage.read().unwrap().affine(&self.layout, 1.0, -rhs)?;
            let op = BackpropOp::none();
            Ok(from_storage(storage, self.layout.shape().clone(), op, false))
        }
    }
    impl std::ops::Mul<Tensor> for f64 {
        type Output = Result<Tensor, Box<dyn std::error::Error>>;
        fn mul(self, rhs: Tensor) -> Self::Output {
            rhs * self
        }
    }
    impl std::ops::Mul<f64> for Tensor {
        type Output = Result<Tensor, Box<dyn std::error::Error>>;
        fn mul(self, rhs: f64) -> Self::Output {
            let storage = self.storage.read().unwrap().affine(&self.layout, rhs, 0.0)?;
            let op = BackpropOp::none();
            Ok(from_storage(storage, self.layout.shape().clone(), op, false))
        }
    }
    impl std::ops::Div<Tensor> for f64 {
        type Output = Result<Tensor, Box<dyn std::error::Error>>;
        fn div(self, rhs: Tensor) -> Self::Output {
            rhs.recip()? * self
        }
    }
    impl std::ops::Div<f64> for Tensor {
        type Output = Result<Tensor, Box<dyn std::error::Error>>;
        fn div(self, rhs: f64) -> Self::Output {
            let storage = self
                .storage
                .read()
                .unwrap()
                .affine(&self.layout, 1.0 / rhs, 0.0)?;
            let op = BackpropOp::none();
            Ok(from_storage(storage, self.layout.shape().clone(), op, false))
        }
    }
    fn from_storage<S: Into<Shape>>(
        storage: Storage,
        shape: S,
        op: BackpropOp,
        is_var: bool,
    ) -> Tensor {
        let shape = shape.into();
        let device = storage.device();
        let dtype = storage.dtype();
        let layout = Layout::contiguous(shape);
        let storage = Arc::new(RwLock::new(storage));
        let tensor = Tensor_ {
            id: TensorId::new(),
            storage,
            layout,
            device,
            op,
            dtype,
            is_var,
        };
        Tensor(Arc::new(tensor))
    }
}
mod storage {
    use crate::cpu_backend::CpuStorage;
    use crate::device::Device;
    use crate::dtype::Dtype;
    use crate::layout::Layout;
    use crate::op::{UnaryOpT, BinaryOpT};
    pub enum Storage {
        Cpu(CpuStorage),
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for Storage {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match self {
                Storage::Cpu(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "Cpu",
                        &__self_0,
                    )
                }
            }
        }
    }
    impl Storage {
        pub fn device(&self) -> Device {
            match self {
                Storage::Cpu(_) => Device::Cpu,
            }
        }
        pub fn dtype(&self) -> Dtype {
            match self {
                Self::Cpu(storage) => storage.dtype(),
            }
        }
        pub fn copy_strided_src(
            &self,
            dst: &mut Self,
            dst_offset: usize,
            src_l: &Layout,
        ) -> Result<(), Box<dyn std::error::Error>> {
            match (self, dst) {
                (Self::Cpu(src), Self::Cpu(dst)) => {
                    src.copy_strided_src(dst, dst_offset, src_l);
                    Ok(())
                }
            }
        }
        pub fn matmul(
            &self,
            rhs: &Self,
            bmnk: (usize, usize, usize, usize),
            lhs_layout: &Layout,
            rhs_layout: &Layout,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            match (self, rhs) {
                (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                    let storage = lhs.matmul(rhs, bmnk, lhs_layout, rhs_layout)?;
                    Ok(Storage::Cpu(storage))
                }
            }
        }
        pub fn affine(
            &self,
            layout: &Layout,
            mul: f64,
            add: f64,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            match self {
                Self::Cpu(storage) => {
                    let storage = storage.affine(layout, mul, add)?;
                    Ok(Storage::Cpu(storage))
                }
            }
        }
        pub fn unary_op<B: UnaryOpT>(
            &self,
            layout: &Layout,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            match self {
                Self::Cpu(storage) => {
                    let storage = storage.unary_op::<B>(layout)?;
                    Ok(Storage::Cpu(storage))
                }
            }
        }
        pub fn binary_op<B: BinaryOpT>(
            &self,
            rhs: &Self,
            layout: &Layout,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            match (self, rhs) {
                (Self::Cpu(lhs), Self::Cpu(rhs)) => {
                    let storage = lhs.binary_op::<B>(rhs, layout)?;
                    Ok(Storage::Cpu(storage))
                }
            }
        }
    }
}
mod cpu_backend {
    #![allow(unused_imports)]
    use std::error::Error;
    use std::sync::Mutex;
    use rand::Rng;
    use rayon::prelude::*;
    use crate::shape::Shape;
    use crate::dtype::Dtype;
    use crate::accelerate::dgemm;
    use crate::layout::Layout;
    use crate::strided_index::StridedBlocks;
    use crate::op::UnaryOpT;
    pub enum CpuStorage {
        I64(Vec<i64>),
        F64(Vec<f64>),
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for CpuStorage {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match self {
                CpuStorage::I64(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "I64",
                        &__self_0,
                    )
                }
                CpuStorage::F64(__self_0) => {
                    ::core::fmt::Formatter::debug_tuple_field1_finish(
                        f,
                        "F64",
                        &__self_0,
                    )
                }
            }
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for CpuStorage {
        #[inline]
        fn clone(&self) -> CpuStorage {
            match self {
                CpuStorage::I64(__self_0) => {
                    CpuStorage::I64(::core::clone::Clone::clone(__self_0))
                }
                CpuStorage::F64(__self_0) => {
                    CpuStorage::F64(::core::clone::Clone::clone(__self_0))
                }
            }
        }
    }
    impl From<Vec<f64>> for CpuStorage {
        fn from(data: Vec<f64>) -> Self {
            CpuStorage::F64(data)
        }
    }
    impl CpuStorage {
        pub fn ones(shape: &Shape, dtype: &Dtype) -> Result<CpuStorage, Box<dyn Error>> {
            let elements = shape.elem_count();
            let storage = match dtype {
                Dtype::I64 => {
                    let data = ::alloc::vec::from_elem(1i64, elements);
                    CpuStorage::I64(data)
                }
                Dtype::F64 => {
                    let data = ::alloc::vec::from_elem(1f64, elements);
                    CpuStorage::F64(data)
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            };
            Ok(storage)
        }
        pub fn zeros(
            shape: &Shape,
            dtype: &Dtype,
        ) -> Result<CpuStorage, Box<dyn Error>> {
            let elements = shape.elem_count();
            let storage = match dtype {
                Dtype::I64 => {
                    let data = ::alloc::vec::from_elem(0i64, elements);
                    CpuStorage::I64(data)
                }
                Dtype::F64 => {
                    let data = ::alloc::vec::from_elem(0f64, elements);
                    CpuStorage::F64(data)
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            };
            Ok(storage)
        }
        pub fn copy_strided_src(
            &self,
            dst: &mut Self,
            dst_offset: usize,
            src_l: &Layout,
        ) {
            match (self, dst) {
                (CpuStorage::F64(src), CpuStorage::F64(dst)) => {
                    match src_l.strided_blocks() {
                        StridedBlocks::SingleBlock { start_offset, len } => {
                            let to_copy = (dst.len() - dst_offset).min(len);
                            dst[dst_offset..dst_offset + to_copy]
                                .copy_from_slice(
                                    &src[start_offset..start_offset + to_copy],
                                );
                        }
                        StridedBlocks::MultipleBlocks {
                            block_start_index,
                            block_len: 1,
                        } => {
                            for (dst_index, src_index) in block_start_index.enumerate() {
                                let dst_index = dst_index + dst_offset;
                                if dst_index >= dst.len() {
                                    break;
                                }
                                dst[dst_index] = src[src_index];
                            }
                        }
                        StridedBlocks::MultipleBlocks {
                            block_start_index,
                            block_len,
                        } => {
                            ::core::panicking::panic_fmt(
                                format_args!(
                                    "not implemented: {0}", format_args!("Not implemented yet3")
                                ),
                            );
                        }
                    }
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            }
        }
        pub fn rand_uniform(
            lo: f64,
            hi: f64,
            shape: &Shape,
            dtype: &Dtype,
        ) -> Result<CpuStorage, Box<dyn Error>> {
            let elements = shape.elem_count();
            let mut rng = rand::thread_rng();
            let storage = match dtype {
                Dtype::F64 => {
                    let uniform = rand::distributions::Uniform::new(lo, hi);
                    let data = (0..elements).map(|_| rng.sample(uniform)).collect();
                    CpuStorage::F64(data)
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            };
            Ok(storage)
        }
        pub fn randn(
            mean: f64,
            std: f64,
            shape: &Shape,
            dtype: &Dtype,
        ) -> Result<CpuStorage, Box<dyn Error>> {
            let elements = shape.elem_count();
            let mut rng = rand::thread_rng();
            let storage = match dtype {
                Dtype::F64 => {
                    let normal = rand_distr::Normal::new(mean, std)?;
                    let data = (0..elements).map(|_| rng.sample(normal)).collect();
                    CpuStorage::F64(data)
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            };
            Ok(storage)
        }
        pub fn dtype(&self) -> Dtype {
            match self {
                CpuStorage::I64(_) => Dtype::I64,
                CpuStorage::F64(_) => Dtype::F64,
            }
        }
        pub fn from(data: Vec<f64>) -> Result<CpuStorage, Box<dyn Error>> {
            Ok(CpuStorage::F64(data))
        }
        pub fn unary_op<B: UnaryOpT>(
            &self,
            layout: &Layout,
        ) -> Result<CpuStorage, Box<dyn Error>> {
            match self {
                CpuStorage::F64(data) => {
                    let result = unary_map(data, layout, B::f64);
                    Ok(CpuStorage::F64(result))
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            }
        }
        pub fn binary_op<B: UnaryOpT>(
            &self,
            rhs: &CpuStorage,
            lhs_l: &Layout,
            rhs_l: &Layout,
        ) -> Result<CpuStorage, Box<dyn Error>> {
            match (self, rhs) {
                (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => {
                    let result = binary_map(lhs_l, rhs_l, lhs, rhs, B::f64);
                    Ok(CpuStorage::F64(result))
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            }
        }
        pub fn affine(
            &self,
            layout: &Layout,
            mul: f64,
            add: f64,
        ) -> Result<CpuStorage, Box<dyn Error>> {
            match self {
                CpuStorage::F64(data) => {
                    let result = data.iter().map(|x| x * mul + add).collect();
                    Ok(CpuStorage::F64(result))
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            }
        }
        #[cfg(not(feature = "accelerate"))]
        pub fn matmul(
            &self,
            rhs: &CpuStorage,
            bmnk: (usize, usize, usize, usize),
            lhs_layout: &Layout,
            rhs_layout: &Layout,
        ) -> Result<CpuStorage, Box<dyn Error>> {
            match (self, rhs) {
                (CpuStorage::F64(lhs), CpuStorage::F64(rhs)) => {
                    let (b, m, n, k) = bmnk;
                    let result = Mutex::new(::alloc::vec::from_elem(0f64, b * m * n));
                    (0..b)
                        .into_par_iter()
                        .for_each(|bi| {
                            for mi in 0..m {
                                for ni in 0..n {
                                    for ki in 0..k {
                                        let lhs_index = bi * lhs_layout.strides()[0]
                                            + ki * lhs_layout.strides()[1] + mi * k;
                                        let rhs_index = bi * rhs_layout.strides()[0]
                                            + ni * rhs_layout.strides()[1] + ki * n;
                                        let result_index = bi * m * n + mi * n + ni;
                                        let mut result_guard = result.lock().unwrap();
                                        result_guard[result_index]
                                            += lhs[lhs_index] * rhs[rhs_index];
                                    }
                                }
                            }
                        });
                    Ok(CpuStorage::F64(result.into_inner().unwrap()))
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            }
        }
    }
    fn unary_map<T: Copy, U: Copy, F: FnMut(T) -> U>(
        data: &[T],
        layout: &Layout,
        mut op: F,
    ) -> Vec<U> {
        match layout.strided_blocks() {
            StridedBlocks::SingleBlock { start_offset, len } => {
                data[start_offset..start_offset + len].iter().map(|&x| op(x)).collect()
            }
            _ => {
                ::core::panicking::panic_fmt(
                    format_args!(
                        "not implemented: {0}", format_args!("Not implemented yet")
                    ),
                );
            }
        }
    }
    fn binary_map<T: Copy, U: Copy, V: Copy, F: FnMut(T, V) -> U>(
        lhs_l: &Layout,
        rhs_l: &Layout,
        lhs: &[T],
        rhs: &[V],
        mut op: F,
    ) -> Vec<U> {
        match (lhs_l.strided_blocks(), rhs_l.strided_blocks()) {
            (
                StridedBlocks::SingleBlock { start_offset: lhs_start, len: lhs_len },
                StridedBlocks::SingleBlock { start_offset: rhs_start, len: rhs_len },
            ) => {
                let lhs = &lhs[lhs_start..lhs_start + lhs_len];
                let rhs = &rhs[rhs_start..rhs_start + rhs_len];
                lhs.iter().zip(rhs.iter()).map(|(&x, &y)| op(x, y)).collect()
            }
            _ => {
                ::core::panicking::panic_fmt(
                    format_args!(
                        "not implemented: {0}", format_args!("Not implemented yet")
                    ),
                );
            }
        }
    }
}
mod layout {
    use crate::shape::Shape;
    use crate::strided_index::{StridedBlocks, StridedIndex};
    pub struct Layout {
        shape: Shape,
        strides: Vec<usize>,
        start_offset: usize,
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for Layout {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field3_finish(
                f,
                "Layout",
                "shape",
                &self.shape,
                "strides",
                &self.strides,
                "start_offset",
                &&self.start_offset,
            )
        }
    }
    impl Layout {
        pub fn new(shape: Shape, stride: Vec<usize>, start_offset: usize) -> Self {
            Self {
                shape,
                strides: stride,
                start_offset,
            }
        }
        pub fn contiguous_with_offset<S: Into<Shape>>(
            shape: S,
            start_offset: usize,
        ) -> Self {
            let shape = shape.into();
            let stride = shape.stride_contiguous();
            Self::new(shape, stride, start_offset)
        }
        pub fn narrow(
            &self,
            dim: usize,
            start: usize,
            len: usize,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let dims = self.dims();
            if dim >= dims.len() {
                return Err("Dimension out of range".into());
            }
            if start + len > dims[dim] {
                return Err("Narrowing out of range".into());
            }
            let mut dims = dims.to_vec();
            dims[dim] = len;
            Ok(
                Self::new(
                    Shape::from(dims),
                    self.strides.clone(),
                    self.start_offset + start * self.strides[dim],
                ),
            )
        }
        pub fn strided_blocks(&self) -> StridedBlocks {
            let mut block_len = 1;
            let mut contiguous_dims = 0;
            for (&stride, &dim) in self
                .strides
                .iter()
                .zip(self.shape.dims().iter())
                .rev()
            {
                if stride != block_len {
                    break;
                }
                block_len *= dim;
                contiguous_dims += 1;
            }
            let index_dims = self.shape.dims().len() - contiguous_dims;
            if index_dims == 0 {
                StridedBlocks::SingleBlock {
                    start_offset: self.start_offset,
                    len: block_len,
                }
            } else {
                let block_start_index = StridedIndex::new(
                    &self.dims()[..index_dims],
                    &self.strides[..index_dims],
                    self.start_offset,
                );
                StridedBlocks::MultipleBlocks {
                    block_start_index,
                    block_len,
                }
            }
        }
        pub fn contiguous<S: Into<Shape>>(shape: S) -> Self {
            Self::contiguous_with_offset(shape, 0)
        }
        pub fn is_contiguous(&self) -> bool {
            self.shape.is_contiguous(&self.strides)
        }
        pub fn is_fortran_contiguous(&self) -> bool {
            self.shape.is_f_contiguous(&self.strides)
        }
        pub fn strides(&self) -> &[usize] {
            &self.strides
        }
        pub fn shape(&self) -> &Shape {
            &self.shape
        }
        pub fn start_offset(&self) -> usize {
            self.start_offset
        }
        pub fn dims(&self) -> &[usize] {
            self.shape.dims()
        }
        pub fn transpose(
            &self,
            dim1: usize,
            dim2: usize,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let rank = self.shape.rank();
            if rank <= dim1 || rank <= dim2 {
                return Err("Dimension out of range".into());
            }
            let mut strides = self.strides.to_vec();
            strides.swap(dim1, dim2);
            let mut dims = self.shape.dims().to_vec();
            dims.swap(dim1, dim2);
            Ok(Self::new(Shape::from(dims), strides, self.start_offset))
        }
    }
}
mod dtype {
    pub enum Dtype {
        I64,
        F64,
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for Dtype {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::write_str(
                f,
                match self {
                    Dtype::I64 => "I64",
                    Dtype::F64 => "F64",
                },
            )
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for Dtype {
        #[inline]
        fn clone(&self) -> Dtype {
            *self
        }
    }
    #[automatically_derived]
    impl ::core::marker::Copy for Dtype {}
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for Dtype {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for Dtype {
        #[inline]
        fn eq(&self, other: &Dtype) -> bool {
            let __self_tag = ::core::intrinsics::discriminant_value(self);
            let __arg1_tag = ::core::intrinsics::discriminant_value(other);
            __self_tag == __arg1_tag
        }
    }
    #[automatically_derived]
    impl ::core::cmp::Eq for Dtype {
        #[inline]
        #[doc(hidden)]
        #[coverage(off)]
        fn assert_receiver_is_total_eq(&self) -> () {}
    }
    #[automatically_derived]
    impl ::core::hash::Hash for Dtype {
        #[inline]
        fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) -> () {
            let __self_tag = ::core::intrinsics::discriminant_value(self);
            ::core::hash::Hash::hash(&__self_tag, state)
        }
    }
}
mod device {
    use std::error::Error;
    use crate::shape::Shape;
    use crate::dtype::Dtype;
    use crate::storage::Storage;
    use crate::cpu_backend::CpuStorage;
    pub enum Device {
        Cpu,
        Gpu,
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for Device {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::write_str(
                f,
                match self {
                    Device::Cpu => "Cpu",
                    Device::Gpu => "Gpu",
                },
            )
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for Device {
        #[inline]
        fn clone(&self) -> Device {
            *self
        }
    }
    #[automatically_derived]
    impl ::core::marker::Copy for Device {}
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for Device {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for Device {
        #[inline]
        fn eq(&self, other: &Device) -> bool {
            let __self_tag = ::core::intrinsics::discriminant_value(self);
            let __arg1_tag = ::core::intrinsics::discriminant_value(other);
            __self_tag == __arg1_tag
        }
    }
    #[automatically_derived]
    impl ::core::cmp::Eq for Device {
        #[inline]
        #[doc(hidden)]
        #[coverage(off)]
        fn assert_receiver_is_total_eq(&self) -> () {}
    }
    #[automatically_derived]
    impl ::core::cmp::PartialOrd for Device {
        #[inline]
        fn partial_cmp(
            &self,
            other: &Device,
        ) -> ::core::option::Option<::core::cmp::Ordering> {
            let __self_tag = ::core::intrinsics::discriminant_value(self);
            let __arg1_tag = ::core::intrinsics::discriminant_value(other);
            ::core::cmp::PartialOrd::partial_cmp(&__self_tag, &__arg1_tag)
        }
    }
    #[automatically_derived]
    impl ::core::hash::Hash for Device {
        #[inline]
        fn hash<__H: ::core::hash::Hasher>(&self, state: &mut __H) -> () {
            let __self_tag = ::core::intrinsics::discriminant_value(self);
            ::core::hash::Hash::hash(&__self_tag, state)
        }
    }
    pub trait NdArray {
        fn shape(&self) -> Result<Shape, Box<dyn Error>>;
        fn to_cpu_storage(&self) -> CpuStorage;
    }
    impl NdArray for f64 {
        fn shape(&self) -> Result<Shape, Box<dyn Error>> {
            Ok(Shape::from(()))
        }
        fn to_cpu_storage(&self) -> CpuStorage {
            CpuStorage::F64(
                <[_]>::into_vec(#[rustc_box] ::alloc::boxed::Box::new([*self])),
            )
        }
    }
    impl NdArray for &[f64] {
        fn shape(&self) -> Result<Shape, Box<dyn Error>> {
            Ok(Shape::from(self.len()))
        }
        fn to_cpu_storage(&self) -> CpuStorage {
            CpuStorage::F64(self.to_vec())
        }
    }
    impl NdArray for Vec<f64> {
        fn shape(&self) -> Result<Shape, Box<dyn Error>> {
            Ok(Shape::from(self.len()))
        }
        fn to_cpu_storage(&self) -> CpuStorage {
            CpuStorage::F64(self.to_vec())
        }
    }
    impl<const N: usize> NdArray for &[f64; N] {
        fn shape(&self) -> Result<Shape, Box<dyn Error>> {
            Ok(Shape::from(N))
        }
        fn to_cpu_storage(&self) -> CpuStorage {
            CpuStorage::F64(self.to_vec())
        }
    }
    impl<const N: usize, const M: usize> NdArray for &[[f64; M]; N] {
        fn shape(&self) -> Result<Shape, Box<dyn Error>> {
            Ok(Shape::from((N, M)))
        }
        fn to_cpu_storage(&self) -> CpuStorage {
            CpuStorage::F64(self.iter().flatten().copied().collect())
        }
    }
    impl<const N1: usize, const N2: usize, const N3: usize> NdArray
    for &[[[f64; N3]; N2]; N1] {
        fn shape(&self) -> Result<Shape, Box<dyn Error>> {
            Ok(Shape::from((N1, N2, N3)))
        }
        fn to_cpu_storage(&self) -> CpuStorage {
            CpuStorage::F64(self.iter().flatten().flatten().copied().collect())
        }
    }
    impl Device {
        fn is_cpu(&self) -> bool {
            match self {
                Device::Cpu => true,
                _ => false,
            }
        }
        fn is_gpu(&self) -> bool {
            match self {
                Device::Gpu => true,
                _ => false,
            }
        }
        pub fn ones(
            &self,
            shape: &Shape,
            dtype: &Dtype,
        ) -> Result<Storage, Box<dyn Error>> {
            match self {
                Device::Cpu => {
                    let storage = CpuStorage::ones(shape, dtype)?;
                    Ok(Storage::Cpu(storage))
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            }
        }
        pub fn zeros(
            &self,
            shape: &Shape,
            dtype: &Dtype,
        ) -> Result<Storage, Box<dyn Error>> {
            match self {
                Device::Cpu => {
                    let storage = CpuStorage::zeros(shape, dtype)?;
                    Ok(Storage::Cpu(storage))
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            }
        }
        pub fn rand_uniform(
            &self,
            lo: f64,
            hi: f64,
            shape: &Shape,
            dtype: &Dtype,
        ) -> Result<Storage, Box<dyn Error>> {
            match self {
                Device::Cpu => {
                    let storage = CpuStorage::rand_uniform(lo, hi, shape, dtype)?;
                    Ok(Storage::Cpu(storage))
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            }
        }
        pub fn randn(
            &self,
            mean: f64,
            std: f64,
            shape: &Shape,
            dtype: &Dtype,
        ) -> Result<Storage, Box<dyn Error>> {
            match self {
                Device::Cpu => {
                    let storage = CpuStorage::randn(mean, std, shape, dtype)?;
                    Ok(Storage::Cpu(storage))
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            }
        }
        pub fn storage_owned(&self, data: Vec<f64>) -> Result<Storage, Box<dyn Error>> {
            match self {
                Device::Cpu => {
                    let storage = CpuStorage::from(data)?;
                    Ok(Storage::Cpu(storage))
                }
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            }
        }
        pub fn storage<A: NdArray>(&self, data: A) -> Result<Storage, Box<dyn Error>> {
            match self {
                Device::Cpu => Ok(Storage::Cpu(data.to_cpu_storage())),
                _ => {
                    ::core::panicking::panic_fmt(
                        format_args!(
                            "not implemented: {0}", format_args!("Not implemented yet")
                        ),
                    );
                }
            }
        }
    }
}
mod op {
    use crate::tensor::Tensor;
    pub enum CmpOp {
        Eq,
        Ne,
        Lt,
        Le,
        Gt,
        Ge,
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for CmpOp {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::write_str(
                f,
                match self {
                    CmpOp::Eq => "Eq",
                    CmpOp::Ne => "Ne",
                    CmpOp::Lt => "Lt",
                    CmpOp::Le => "Le",
                    CmpOp::Gt => "Gt",
                    CmpOp::Ge => "Ge",
                },
            )
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for CmpOp {
        #[inline]
        fn clone(&self) -> CmpOp {
            *self
        }
    }
    #[automatically_derived]
    impl ::core::marker::Copy for CmpOp {}
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for CmpOp {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for CmpOp {
        #[inline]
        fn eq(&self, other: &CmpOp) -> bool {
            let __self_tag = ::core::intrinsics::discriminant_value(self);
            let __arg1_tag = ::core::intrinsics::discriminant_value(other);
            __self_tag == __arg1_tag
        }
    }
    #[automatically_derived]
    impl ::core::cmp::Eq for CmpOp {
        #[inline]
        #[doc(hidden)]
        #[coverage(off)]
        fn assert_receiver_is_total_eq(&self) -> () {}
    }
    pub enum Op {
        Unary(Tensor, UnaryOp),
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for Op {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match self {
                Op::Unary(__self_0, __self_1) => {
                    ::core::fmt::Formatter::debug_tuple_field2_finish(
                        f,
                        "Unary",
                        __self_0,
                        &__self_1,
                    )
                }
            }
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for Op {
        #[inline]
        fn clone(&self) -> Op {
            match self {
                Op::Unary(__self_0, __self_1) => {
                    Op::Unary(
                        ::core::clone::Clone::clone(__self_0),
                        ::core::clone::Clone::clone(__self_1),
                    )
                }
            }
        }
    }
    pub enum UnaryOp {
        Reciprocal,
        Abs,
        Neg,
        Exp,
        Relu,
        Sqr,
        Log,
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for UnaryOp {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::write_str(
                f,
                match self {
                    UnaryOp::Reciprocal => "Reciprocal",
                    UnaryOp::Abs => "Abs",
                    UnaryOp::Neg => "Neg",
                    UnaryOp::Exp => "Exp",
                    UnaryOp::Relu => "Relu",
                    UnaryOp::Sqr => "Sqr",
                    UnaryOp::Log => "Log",
                },
            )
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for UnaryOp {
        #[inline]
        fn clone(&self) -> UnaryOp {
            *self
        }
    }
    #[automatically_derived]
    impl ::core::marker::Copy for UnaryOp {}
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for UnaryOp {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for UnaryOp {
        #[inline]
        fn eq(&self, other: &UnaryOp) -> bool {
            let __self_tag = ::core::intrinsics::discriminant_value(self);
            let __arg1_tag = ::core::intrinsics::discriminant_value(other);
            __self_tag == __arg1_tag
        }
    }
    #[automatically_derived]
    impl ::core::cmp::Eq for UnaryOp {
        #[inline]
        #[doc(hidden)]
        #[coverage(off)]
        fn assert_receiver_is_total_eq(&self) -> () {}
    }
    pub enum BinaryOp {
        Add,
        Sub,
        Mul,
        Div,
        Pow,
        Max,
        Min,
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for BinaryOp {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::write_str(
                f,
                match self {
                    BinaryOp::Add => "Add",
                    BinaryOp::Sub => "Sub",
                    BinaryOp::Mul => "Mul",
                    BinaryOp::Div => "Div",
                    BinaryOp::Pow => "Pow",
                    BinaryOp::Max => "Max",
                    BinaryOp::Min => "Min",
                },
            )
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for BinaryOp {
        #[inline]
        fn clone(&self) -> BinaryOp {
            *self
        }
    }
    #[automatically_derived]
    impl ::core::marker::Copy for BinaryOp {}
    #[automatically_derived]
    impl ::core::marker::StructuralPartialEq for BinaryOp {}
    #[automatically_derived]
    impl ::core::cmp::PartialEq for BinaryOp {
        #[inline]
        fn eq(&self, other: &BinaryOp) -> bool {
            let __self_tag = ::core::intrinsics::discriminant_value(self);
            let __arg1_tag = ::core::intrinsics::discriminant_value(other);
            __self_tag == __arg1_tag
        }
    }
    #[automatically_derived]
    impl ::core::cmp::Eq for BinaryOp {
        #[inline]
        #[doc(hidden)]
        #[coverage(off)]
        fn assert_receiver_is_total_eq(&self) -> () {}
    }
    pub trait UnaryOpT {
        const NAME: &'static str;
        const V: Self;
        fn f64(v1: f64) -> f64;
    }
    pub trait BinaryOpT {
        const NAME: &'static str;
        const V: Self;
        fn f64(v1: f64, v2: f64) -> f64;
    }
    pub struct Reciprocal;
    pub struct Abs;
    pub struct Neg;
    pub struct Exp;
    pub struct Relu;
    pub struct Sqr;
    pub struct Log;
    pub struct Add;
    pub struct Mul;
    pub struct Sub;
    pub struct Div;
    impl BinaryOpT for Add {
        const NAME: &'static str = "add";
        const V: Self = Add;
        #[inline(always)]
        fn f64(v1: f64, v2: f64) -> f64 {
            v1 + v2
        }
    }
    impl BinaryOpT for Mul {
        const NAME: &'static str = "mul";
        const V: Self = Mul;
        #[inline(always)]
        fn f64(v1: f64, v2: f64) -> f64 {
            v1 * v2
        }
    }
    impl BinaryOpT for Sub {
        const NAME: &'static str = "sub";
        const V: Self = Sub;
        #[inline(always)]
        fn f64(v1: f64, v2: f64) -> f64 {
            v1 - v2
        }
    }
    impl BinaryOpT for Div {
        const NAME: &'static str = "div";
        const V: Self = Div;
        #[inline(always)]
        fn f64(v1: f64, v2: f64) -> f64 {
            v1 / v2
        }
    }
    impl UnaryOpT for Reciprocal {
        const NAME: &'static str = "reciprocal";
        const V: Self = Reciprocal;
        #[inline(always)]
        fn f64(v1: f64) -> f64 {
            v1.recip()
        }
    }
    impl UnaryOpT for Abs {
        const NAME: &'static str = "abs";
        const V: Self = Abs;
        #[inline(always)]
        fn f64(v1: f64) -> f64 {
            v1.abs()
        }
    }
    impl UnaryOpT for Neg {
        const NAME: &'static str = "neg";
        const V: Self = Neg;
        #[inline(always)]
        fn f64(v1: f64) -> f64 {
            -v1
        }
    }
    impl UnaryOpT for Exp {
        const NAME: &'static str = "exp";
        const V: Self = Exp;
        #[inline(always)]
        fn f64(v1: f64) -> f64 {
            v1.exp()
        }
    }
    impl UnaryOpT for Relu {
        const NAME: &'static str = "relu";
        const V: Self = Relu;
        #[inline(always)]
        fn f64(v1: f64) -> f64 {
            if v1 > 0.0 { v1 } else { 0.0 }
        }
    }
    impl UnaryOpT for Sqr {
        const NAME: &'static str = "sqr";
        const V: Self = Sqr;
        #[inline(always)]
        fn f64(v1: f64) -> f64 {
            v1 * v1
        }
    }
    impl UnaryOpT for Log {
        const NAME: &'static str = "log";
        const V: Self = Log;
        #[inline(always)]
        fn f64(v1: f64) -> f64 {
            v1.ln()
        }
    }
    pub struct BackpropOp(Option<Op>);
    #[automatically_derived]
    impl ::core::fmt::Debug for BackpropOp {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_tuple_field1_finish(f, "BackpropOp", &&self.0)
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for BackpropOp {
        #[inline]
        fn clone(&self) -> BackpropOp {
            BackpropOp(::core::clone::Clone::clone(&self.0))
        }
    }
    impl BackpropOp {
        pub fn new1(arg: &Tensor, f: impl Fn(Tensor) -> Op) -> Self {
            if arg.track_op() { Self(Some(f(arg.clone()))) } else { Self(None) }
        }
        pub fn none() -> Self {
            Self(None)
        }
    }
    impl std::ops::Deref for BackpropOp {
        type Target = Option<Op>;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
}
mod accelerate {
    extern crate libc;
    use libc::{c_char, c_double, c_float, c_int};
    mod ffi {
        use super::*;
        extern "C" {
            #[link_name = "dgemm_"]
            pub fn dgemm_ffi(
                transa: *const c_char,
                transb: *const c_char,
                m: *const c_int,
                n: *const c_int,
                k: *const c_int,
                alpha: *const c_double,
                a: *const c_double,
                lda: *const c_int,
                b: *const c_double,
                ldb: *const c_int,
                beta: *const c_double,
                c: *mut c_double,
                ldc: *const c_int,
            );
            pub fn srotg_(
                a: *mut c_float,
                b: *mut c_float,
                c: *mut c_float,
                s: *mut c_float,
            );
        }
    }
    #[inline]
    pub unsafe fn dgemm(
        transa: u8,
        transb: u8,
        m: i32,
        n: i32,
        k: i32,
        alpha: f64,
        a: &[f64],
        lda: i32,
        b: &[f64],
        ldb: i32,
        beta: f64,
        c: &mut [f64],
        ldc: i32,
    ) {
        ffi::dgemm_ffi(
            &(transa as c_char),
            &(transb as c_char),
            &m,
            &n,
            &k,
            &alpha,
            a.as_ptr(),
            &lda,
            b.as_ptr(),
            &ldb,
            &beta,
            c.as_mut_ptr(),
            &ldc,
        )
    }
}
mod strided_index {
    pub enum StridedBlocks<'a> {
        SingleBlock { start_offset: usize, len: usize },
        MultipleBlocks { block_start_index: StridedIndex<'a>, block_len: usize },
    }
    #[automatically_derived]
    impl<'a> ::core::fmt::Debug for StridedBlocks<'a> {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match self {
                StridedBlocks::SingleBlock { start_offset: __self_0, len: __self_1 } => {
                    ::core::fmt::Formatter::debug_struct_field2_finish(
                        f,
                        "SingleBlock",
                        "start_offset",
                        __self_0,
                        "len",
                        &__self_1,
                    )
                }
                StridedBlocks::MultipleBlocks {
                    block_start_index: __self_0,
                    block_len: __self_1,
                } => {
                    ::core::fmt::Formatter::debug_struct_field2_finish(
                        f,
                        "MultipleBlocks",
                        "block_start_index",
                        __self_0,
                        "block_len",
                        &__self_1,
                    )
                }
            }
        }
    }
    #[automatically_derived]
    impl<'a> ::core::clone::Clone for StridedBlocks<'a> {
        #[inline]
        fn clone(&self) -> StridedBlocks<'a> {
            match self {
                StridedBlocks::SingleBlock { start_offset: __self_0, len: __self_1 } => {
                    StridedBlocks::SingleBlock {
                        start_offset: ::core::clone::Clone::clone(__self_0),
                        len: ::core::clone::Clone::clone(__self_1),
                    }
                }
                StridedBlocks::MultipleBlocks {
                    block_start_index: __self_0,
                    block_len: __self_1,
                } => {
                    StridedBlocks::MultipleBlocks {
                        block_start_index: ::core::clone::Clone::clone(__self_0),
                        block_len: ::core::clone::Clone::clone(__self_1),
                    }
                }
            }
        }
    }
    #[automatically_derived]
    impl<'a> ::core::marker::StructuralPartialEq for StridedBlocks<'a> {}
    #[automatically_derived]
    impl<'a> ::core::cmp::PartialEq for StridedBlocks<'a> {
        #[inline]
        fn eq(&self, other: &StridedBlocks<'a>) -> bool {
            let __self_tag = ::core::intrinsics::discriminant_value(self);
            let __arg1_tag = ::core::intrinsics::discriminant_value(other);
            __self_tag == __arg1_tag
                && match (self, other) {
                    (
                        StridedBlocks::SingleBlock {
                            start_offset: __self_0,
                            len: __self_1,
                        },
                        StridedBlocks::SingleBlock {
                            start_offset: __arg1_0,
                            len: __arg1_1,
                        },
                    ) => *__self_0 == *__arg1_0 && *__self_1 == *__arg1_1,
                    (
                        StridedBlocks::MultipleBlocks {
                            block_start_index: __self_0,
                            block_len: __self_1,
                        },
                        StridedBlocks::MultipleBlocks {
                            block_start_index: __arg1_0,
                            block_len: __arg1_1,
                        },
                    ) => *__self_0 == *__arg1_0 && *__self_1 == *__arg1_1,
                    _ => unsafe { ::core::intrinsics::unreachable() }
                }
        }
    }
    #[automatically_derived]
    impl<'a> ::core::cmp::Eq for StridedBlocks<'a> {
        #[inline]
        #[doc(hidden)]
        #[coverage(off)]
        fn assert_receiver_is_total_eq(&self) -> () {
            let _: ::core::cmp::AssertParamIsEq<usize>;
            let _: ::core::cmp::AssertParamIsEq<StridedIndex<'a>>;
        }
    }
    pub struct StridedIndex<'a> {
        next_storage_index: Option<usize>,
        multi_index: Vec<usize>,
        dims: &'a [usize],
        stride: &'a [usize],
    }
    #[automatically_derived]
    impl<'a> ::core::fmt::Debug for StridedIndex<'a> {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_struct_field4_finish(
                f,
                "StridedIndex",
                "next_storage_index",
                &self.next_storage_index,
                "multi_index",
                &self.multi_index,
                "dims",
                &self.dims,
                "stride",
                &&self.stride,
            )
        }
    }
    #[automatically_derived]
    impl<'a> ::core::clone::Clone for StridedIndex<'a> {
        #[inline]
        fn clone(&self) -> StridedIndex<'a> {
            StridedIndex {
                next_storage_index: ::core::clone::Clone::clone(
                    &self.next_storage_index,
                ),
                multi_index: ::core::clone::Clone::clone(&self.multi_index),
                dims: ::core::clone::Clone::clone(&self.dims),
                stride: ::core::clone::Clone::clone(&self.stride),
            }
        }
    }
    #[automatically_derived]
    impl<'a> ::core::marker::StructuralPartialEq for StridedIndex<'a> {}
    #[automatically_derived]
    impl<'a> ::core::cmp::PartialEq for StridedIndex<'a> {
        #[inline]
        fn eq(&self, other: &StridedIndex<'a>) -> bool {
            self.next_storage_index == other.next_storage_index
                && self.multi_index == other.multi_index && self.dims == other.dims
                && self.stride == other.stride
        }
    }
    #[automatically_derived]
    impl<'a> ::core::cmp::Eq for StridedIndex<'a> {
        #[inline]
        #[doc(hidden)]
        #[coverage(off)]
        fn assert_receiver_is_total_eq(&self) -> () {
            let _: ::core::cmp::AssertParamIsEq<Option<usize>>;
            let _: ::core::cmp::AssertParamIsEq<Vec<usize>>;
            let _: ::core::cmp::AssertParamIsEq<&'a [usize]>;
            let _: ::core::cmp::AssertParamIsEq<&'a [usize]>;
        }
    }
    impl<'a> StridedIndex<'a> {
        pub fn new(dims: &'a [usize], stride: &'a [usize], start_offset: usize) -> Self {
            let elem_count: usize = dims.iter().product();
            let next_storage_index = if elem_count == 0 {
                None
            } else {
                Some(start_offset)
            };
            StridedIndex {
                next_storage_index,
                multi_index: ::alloc::vec::from_elem(0, dims.len()),
                dims,
                stride,
            }
        }
    }
    impl<'a> Iterator for StridedIndex<'a> {
        type Item = usize;
        fn next(&mut self) -> Option<Self::Item> {
            let storage_index = match self.next_storage_index {
                None => return None,
                Some(storage_index) => storage_index,
            };
            let mut updated = false;
            let mut next_storage_index = storage_index;
            for ((multi_i, max_i), stride_i) in self
                .multi_index
                .iter_mut()
                .zip(self.dims.iter())
                .zip(self.stride.iter())
                .rev()
            {
                let next_i = *multi_i + 1;
                if next_i < *max_i {
                    *multi_i = next_i;
                    updated = true;
                    next_storage_index += stride_i;
                    break;
                } else {
                    next_storage_index -= *multi_i * stride_i;
                    *multi_i = 0;
                }
            }
            self
                .next_storage_index = if updated {
                Some(next_storage_index)
            } else {
                None
            };
            Some(storage_index)
        }
    }
}
mod backprop {
    use std::collections::HashMap;
    use std::ops::Mul;
    use crate::tensor::{Tensor, TensorId};
    use crate::op::*;
    use crate::variables::Var;
    impl Tensor {
        fn sorted_nodes(&self) -> Vec<&Tensor> {
            fn walk<'a>(
                node: &'a Tensor,
                nodes: Vec<&'a Tensor>,
                already_seen: &mut HashMap<TensorId, bool>,
            ) -> (bool, Vec<&'a Tensor>) {
                if let Some(&tg) = already_seen.get(&node.id()) {
                    return (tg, nodes);
                }
                let mut track_grad = false;
                let mut nodes = if node.is_var() {
                    track_grad = true;
                    nodes
                } else if let Some(op) = node.op() {
                    match op {
                        Op::Unary(node, _) => {
                            let (tg, mut nodes) = walk(node, nodes, already_seen);
                            track_grad |= tg;
                            nodes
                        }
                        _ => {
                            ::core::panicking::panic_fmt(
                                format_args!(
                                    "not implemented: {0}", format_args!("Op not implemented")
                                ),
                            );
                        }
                    }
                } else {
                    nodes
                };
                already_seen.insert(node.id(), track_grad);
                if track_grad {
                    nodes.push(node);
                }
                (track_grad, nodes)
            }
            let (_tg, mut nodes) = walk(
                self,
                ::alloc::vec::Vec::new(),
                &mut HashMap::new(),
            );
            nodes.reverse();
            nodes
        }
        pub fn backward(&self) -> Result<GradStore, Box<dyn std::error::Error>> {
            let nodes = self.sorted_nodes();
            let mut grads = GradStore::new();
            grads.insert(self, Tensor::ones_like(self)?.contiguous().unwrap());
            for node in nodes.iter() {
                if node.is_var() {
                    continue;
                }
                let grad = grads.remove(node).expect("Node not found in grads");
                {
                    ::std::io::_print(format_args!("grad: {0}\n", grad));
                };
                if let Some(op) = node.op() {
                    match op {
                        Op::Unary(a, UnaryOp::Sqr) => {
                            let grad = a.mul(&grad)?.mul(&Tensor::from(2.0))?;
                            grads.insert(a, grad);
                        }
                        _ => {
                            ::core::panicking::panic_fmt(
                                format_args!(
                                    "not implemented: {0}", format_args!("Op not implemented")
                                ),
                            );
                        }
                    }
                }
            }
            Ok(grads)
        }
    }
    pub struct GradStore(HashMap<TensorId, Tensor>);
    #[automatically_derived]
    impl ::core::fmt::Debug for GradStore {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_tuple_field1_finish(f, "GradStore", &&self.0)
        }
    }
    impl GradStore {
        pub fn new() -> Self {
            Self(HashMap::new())
        }
        pub fn get_id(&self, id: TensorId) -> Option<&Tensor> {
            self.0.get(&id)
        }
        pub fn get(&self, tensor: &Tensor) -> Option<&Tensor> {
            self.0.get(&tensor.id())
        }
        pub fn remove(&mut self, tensor: &Tensor) -> Option<Tensor> {
            self.0.remove(&tensor.id())
        }
        pub fn insert(&mut self, tensor: &Tensor, grad: Tensor) -> Option<Tensor> {
            self.0.insert(tensor.id(), grad)
        }
    }
}
mod indexer {
    use std::ops::*;
    use crate::tensor::Tensor;
    pub enum TensorIndexer {
        Narrow(Bound<usize>, Bound<usize>),
    }
    #[automatically_derived]
    impl ::core::fmt::Debug for TensorIndexer {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            match self {
                TensorIndexer::Narrow(__self_0, __self_1) => {
                    ::core::fmt::Formatter::debug_tuple_field2_finish(
                        f,
                        "Narrow",
                        __self_0,
                        &__self_1,
                    )
                }
            }
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for TensorIndexer {
        #[inline]
        fn clone(&self) -> TensorIndexer {
            match self {
                TensorIndexer::Narrow(__self_0, __self_1) => {
                    TensorIndexer::Narrow(
                        ::core::clone::Clone::clone(__self_0),
                        ::core::clone::Clone::clone(__self_1),
                    )
                }
            }
        }
    }
    pub trait RB: RangeBounds<usize> {}
    impl RB for Range<usize> {}
    impl RB for RangeInclusive<usize> {}
    impl RB for RangeFrom<usize> {}
    impl RB for RangeToInclusive<usize> {}
    impl RB for RangeTo<usize> {}
    impl<T: RB> From<T> for TensorIndexer {
        fn from(range: T) -> Self {
            let start = match range.start_bound() {
                Bound::Included(idx) => Bound::Included(*idx),
                Bound::Excluded(idx) => Bound::Excluded(*idx),
                Bound::Unbounded => Bound::Unbounded,
            };
            let end = match range.end_bound() {
                Bound::Included(idx) => Bound::Included(*idx),
                Bound::Excluded(idx) => Bound::Excluded(*idx),
                Bound::Unbounded => Bound::Unbounded,
            };
            TensorIndexer::Narrow(start, end)
        }
    }
    pub trait IndexOp<T> {
        fn i(&self, indexers: T) -> Result<Tensor, Box<dyn std::error::Error>>;
    }
    impl<T> IndexOp<T> for Tensor
    where
        T: Into<TensorIndexer>,
    {
        fn i(&self, indexers: T) -> Result<Tensor, Box<dyn std::error::Error>> {
            let indexers = <[_]>::into_vec(
                #[rustc_box]
                ::alloc::boxed::Box::new([indexers.into()]),
            );
            self.index(&indexers)
        }
    }
    impl Tensor {
        fn index(
            &self,
            indexers: &[TensorIndexer],
        ) -> Result<Tensor, Box<dyn std::error::Error>> {
            let dims = self.shape().dims();
            let mut x = self.clone();
            let mut current_dim = 0;
            for (i, indexer) in indexers.iter().enumerate() {
                match indexer {
                    TensorIndexer::Narrow(start, end) => {
                        let start = match start {
                            Bound::Included(v) => *v,
                            Bound::Excluded(v) => *v + 1,
                            Bound::Unbounded => 0,
                        };
                        let end = match end {
                            Bound::Included(v) => *v + 1,
                            Bound::Excluded(v) => *v,
                            Bound::Unbounded => dims[i],
                        };
                        let len = end - start;
                        x = x.narrow(current_dim, start, len)?;
                        current_dim += 1;
                    }
                }
            }
            Ok(x)
        }
    }
}
mod variables {
    use crate::device::{Device, NdArray};
    use crate::dtype::Dtype;
    use crate::shape::Shape;
    use crate::tensor::Tensor;
    pub struct Var(Tensor);
    #[automatically_derived]
    impl ::core::fmt::Debug for Var {
        #[inline]
        fn fmt(&self, f: &mut ::core::fmt::Formatter) -> ::core::fmt::Result {
            ::core::fmt::Formatter::debug_tuple_field1_finish(f, "Var", &&self.0)
        }
    }
    #[automatically_derived]
    impl ::core::clone::Clone for Var {
        #[inline]
        fn clone(&self) -> Var {
            Var(::core::clone::Clone::clone(&self.0))
        }
    }
    impl std::fmt::Display for Var {
        fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
            std::fmt::Display::fmt(&self.0, f)
        }
    }
    impl std::ops::Deref for Var {
        type Target = Tensor;
        fn deref(&self) -> &Self::Target {
            &self.0
        }
    }
    impl Var {
        pub fn new<A: NdArray>(
            array: A,
            device: &Device,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let inner = Tensor::new_impl(array, device, true)?;
            Ok(Self(inner))
        }
        pub fn zeros<S: Into<Shape>>(
            shape: S,
            dtype: Dtype,
            device: &Device,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let inner = Tensor::zeros_impl(shape, dtype, device, true)?;
            Ok(Self(inner))
        }
        pub fn ones<S: Into<Shape>>(
            shape: S,
            dtype: Dtype,
            device: &Device,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let inner = Tensor::ones_impl(shape, dtype, device, true)?;
            Ok(Self(inner))
        }
        pub fn rand<S: Into<Shape>>(
            mean: f64,
            std: f64,
            s: S,
            device: &Device,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let inner = Tensor::randn_impl(mean, std, s, device, true)?;
            Ok(Self(inner))
        }
        pub fn randn_uniform<S: Into<Shape>>(
            lo: f64,
            up: f64,
            s: S,
            device: &Device,
        ) -> Result<Self, Box<dyn std::error::Error>> {
            let inner = Tensor::rand_uniform_impl(lo, up, s, device, true)?;
            Ok(Self(inner))
        }
        pub fn from_tensor(t: &Tensor) -> Self {
            Self(t.clone())
        }
    }
}
pub fn add(left: usize, right: usize) -> usize {
    left + right
}
