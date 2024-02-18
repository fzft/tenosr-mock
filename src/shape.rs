use std::error::Error;
use std::fmt::Debug;
use std::boxed::Box;

#[derive(Clone, PartialEq, Eq)]
pub struct Shape(Vec<usize>);

pub const SCALAR: Shape = Shape(vec![]);

impl Debug for Shape {
    fn fmt(&self, f: &mut std::fmt::Formatter) -> std::fmt::Result {
        write!(f, "Shape({:?})", self.0)
    }
}

impl <const C: usize> From<[usize; C]> for Shape {
    fn from(shape: [usize; C]) -> Self {
        Self(shape.to_vec())
    }
}

impl <const C: usize> From<&[usize; C]> for Shape {
    fn from(shape: &[usize; C]) -> Self {
        Self(shape.to_vec())
    }

}

impl From <Vec<usize>> for Shape {
    fn from(shape: Vec<usize>) -> Self {
        Self(shape)
    }
}

impl From <&[usize]> for Shape {
    fn from(shape: &[usize]) -> Self {
        Self(shape.to_vec())
    }
}

impl From <(usize, usize)> for Shape {
    fn from(shape: (usize, usize)) -> Self {
        Self(vec![shape.0, shape.1])
    }
}

impl From<usize> for Shape {
    fn from(shape: usize) -> Self {
        Self(vec![shape])
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
            .scan(1, |prod, u| {
                let prod_pre_mult = *prod;
                *prod *= u;
                Some(prod_pre_mult)
            })
            .collect();
        stride.reverse();
        stride
    }

    // Check if the tensor is C-contiguous
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

    // Check if the tensor is F-contiguous
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

    // Two tensors are “broadcastable” if the following rules hold:
    // 1. Each tensor has at least one dimension.
    // 2.When iterating over the dimension sizes, starting at the trailing dimension, the dimension sizes must either be equal, one of them is 1, or one of them does not exist.
    pub fn broadcast_shape_binary_to(&self, rhs: &Self, op: &'static str) -> Result<Shape, Box<dyn Error>> {
        let lhs = self;
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();
        let lhs_ndims = lhs_dims.len();
        let rhs_ndims = rhs_dims.len();
        let bcast_ndims = usize::max(lhs_ndims, rhs_ndims);
        let mut bcast_dims = vec![0; bcast_ndims];
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
                Err(format!(
                    "Incompatible shapes for broadcasting: {:?} {:?} {:?}",
                    lhs, op, rhs
                ))?;
            }
        }

        bcast_dims.reverse();
        Ok(Shape::from_dims(&bcast_dims))
    }

    /*
    Matrix product of two tensors.
    1. If both tensors are 1-dimensional, the dot product (scalar) is returned.
    2. If both arguments are 2-dimensional, the matrix-matrix product is returned.
    3. If the first argument is 1-dimensional and the second argument is 2-dimensional, a 1 is prepended to its dimension for the purpose of the matrix multiply. After the matrix multiply, the prepended dimension is removed.
    4. If the first argument is 2-dimensional and the second argument is 1-dimensional, the matrix-vector product is returned.
    5. If both arguments are at least 1-dimensional and at least one argument is N-dimensional (where N > 2), then a batched matrix multiply is returned. If the first argument is 1-dimensional, a 1 is prepended to its dimension for the purpose of the batched matrix multiply and removed after. If the second argument is 1-dimensional, a 1 is appended to its dimension for the purpose of the batched matrix multiple and removed after. The non-matrix (i.e. batch) dimensions are broadcasted (and thus must be broadcastable). For example, if input is a
(
�
×
1
×
�
×
�
)
(j×1×n×n) tensor and other is a
(
�
×
�
×
�
)
(k×n×n) tensor, out will be a
(
�
×
�
×
�
×
�
)
(j×k×n×n) tensor.

Note that the broadcasting logic only looks at the batch dimensions when determining if the inputs are broadcastable, and not the matrix dimensions. For example, if input is a
(
�
×
1
×
�
×
�
)
(j×1×n×m) tensor and other is a
(
�
×
�
×
�
)
(k×m×p) tensor, these inputs are valid for broadcasting even though the final two dimensions (i.e. the matrix dimensions) are different. out will be a
(
�
×
�
×
�
×
�
)
(j×k×n×p) tensor.

     */

    pub fn broadcast_shape_matmul(&self, rhs: &Self) -> Result<(Shape, Shape), Box<dyn Error>> {
        let lhs = self;
        let lhs_dims = lhs.dims();
        let rhs_dims = rhs.dims();
        let (m, lhs_k) = (lhs_dims[lhs_dims.len() - 2], lhs_dims[lhs_dims.len() - 1]);
        let (rhs_k, n) = (rhs_dims[rhs_dims.len() - 2], rhs_dims[rhs_dims.len() - 1]);
        if lhs_k != rhs_k {
            Err(format!(
                "Incompatible shapes for matrix multiplication: {:?} {:?}",
                lhs, rhs
            ))?;
        }

        let lhs_b = Self::from(&lhs_dims[..lhs_dims.len() - 2]);
        let rhs_b = Self::from(&rhs_dims[..rhs_dims.len() - 2]);
        let bcast_shape = lhs_b.broadcast_shape_binary_to(&rhs_b, "broadcast_matmul")?;
        let bcast_dims = bcast_shape.dims();

        let bcast_lhs = [bcast_dims, &[m, lhs_k]].concat();
        let bcast_rhs = [bcast_dims, &[rhs_k, n]].concat();
        Ok((Shape::from(bcast_lhs), Shape::from(bcast_rhs)))
    }
}

pub trait ShapeWithOneHole {
    fn into_shape(self, el_count: usize) -> Result<Shape, Box<dyn Error>>;
}

impl <S: Into<Shape>> ShapeWithOneHole for S {
    fn into_shape(self, _el_count: usize) -> Result<Shape, Box<dyn Error>> {
        Ok(self.into())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn shape_new() {
        let shape = Shape::new(vec![1, 2, 3]);
        assert_eq!(shape.0, vec![1, 2, 3]);
    }

    #[test]
    fn shape_len() {
        let shape = Shape::new(vec![1, 2, 3]);
        assert_eq!(shape.len(), 3);
    }

    #[test]
    fn shape_size() {
        let shape = Shape::new(vec![1, 2, 3]);
        assert_eq!(shape.size(), 6);
    }

    #[test]
    fn shape_get() {
        let shape = Shape::new(vec![1, 2, 3]);
        assert_eq!(shape.get(0), Some(&1));
        assert_eq!(shape.get(1), Some(&2));
        assert_eq!(shape.get(2), Some(&3));
        assert_eq!(shape.get(3), None);
    }

    #[test]
    fn shape_as_slice() {
        let shape = Shape::new(vec![1, 2, 3]);
        assert_eq!(shape.as_slice(), &[1, 2, 3]);
    }

    #[test]
    fn shape_from() {
        let shape = Shape::from([1, 2, 3]);
        assert_eq!(shape.0, vec![1, 2, 3]);
    }

    #[test]
    fn shape_from_vec() {
        let shape = Shape::from(vec![1, 2, 3]);
        assert_eq!(shape.0, vec![1, 2, 3]);
    }

    #[test]
    fn shape_from_slice() {
        let shape = Shape::from(&[1, 2, 3]);
        assert_eq!(shape.0, vec![1, 2, 3]);
    }

    #[test]
    fn shape_from_tuple() {
        let shape = Shape::from((1, 2));
        assert_eq!(shape.0, vec![1, 2]);
    }

    #[test]
    fn shape_eq() {
        let shape1 = Shape::from([1, 2, 3]);
        let shape2 = Shape::from([1, 2, 3]);
        assert_eq!(shape1, shape2);
    }

    #[test]
    fn shape_ne() {
        let shape1 = Shape::from([1, 2, 3]);
        let shape2 = Shape::from([1, 2, 4]);
        assert_ne!(shape1, shape2);
    }

    #[test]
    fn shape_scalar() {
        let shape = Shape::from([]);
        assert_eq!(shape, SCALAR);
    }

    #[test]
    fn shape_elem_count() {
        let shape = Shape::from([3, 2, 1]);
        assert_eq!(shape.elem_count(), 6);
    }

    #[test]
    fn shape_stride_contiguous() {
        let shape = Shape::from([3,4]);
        assert_eq!(shape.stride_contiguous(), vec![4, 1]);
    }

    #[test]
    fn shape_stride_contiguous_1() {
        let shape = Shape::from([1000, 2]);
        assert_eq!(shape.stride_contiguous(), vec![2, 1]);
    }

    #[test]
    fn shape_stride_contiguous_2() {
        let shape = Shape::from([1,2,3]);
        assert_eq!(shape.stride_contiguous(), vec![6, 3, 1]);
    }

    #[test]
    fn shape_stride_contiguous_3() {
        let shape = Shape::from([3,2,1]);
        assert_eq!(shape.stride_contiguous(), vec![2, 1, 1]);
    }

    #[test]
    fn shape_is_contiguous() {
        let shape = Shape::from([3,4]);
        let stride = vec![4, 1];
        assert_eq!(shape.is_contiguous(&stride), true);
    }

    #[test]
    fn shape_is_f_contiguous() {
        let shape = Shape::from([3, 4]);
        let stride = vec![1,3];
        assert_eq!(shape.is_f_contiguous(&stride), true);
    }

    #[test]
    fn shape_broadcast_shape_binary_to() {
        let shape1 = Shape::from([3, 4]);
        let shape2 = Shape::from([3, 1]);
        let result = shape1.broadcast_shape_binary_to(&shape2, "add");
        assert_eq!(result.unwrap(), Shape::from([3, 4]));
    }

    #[test]
    fn shape_broadcast_shape_binary_to_2() {
        let shape1 = Shape::from([3, 4]);
        let shape2 = Shape::from([3, 4]);
        let result = shape1.broadcast_shape_binary_to(&shape2, "add");
        assert_eq!(result.unwrap(), Shape::from([3, 4]));
    }

    #[test]
    fn shape_broadcast_shape_binary_to_3() {
        let shape1 = Shape::from([5,3,4,1]);
        let shape2 = Shape::from([3,1,1]);
        let result = shape1.broadcast_shape_binary_to(&shape2, "add");
        assert_eq!(result.unwrap(), Shape::from([5,3,4,1]));
    }

    #[test]
    fn shape_broadcast_shape_binary_to_4() {
        let shape1 = Shape::from([0]);
        let shape2 = Shape::from([2,2]);
        let result = shape2.broadcast_shape_binary_to(&shape1, "add");
        // assert error
        assert_eq!(result.is_err(), true);
    }

    #[test]
    fn shape_broadcast_shape_matmul() {
        let shape1 = Shape::from([3, 4]);
        let shape2 = Shape::from([4, 5]);
        let result = shape1.broadcast_shape_matmul(&shape2);
        assert_eq!(result.unwrap(), (Shape::from([3, 4]), Shape::from([4, 5])));
    }

    #[test]
    fn shape_broadcast_shape_matmul_2() {
        let shape1 = Shape::from([5, 4, 3]);
        let shape2 = Shape::from([3, 4]);
        let result = shape1.broadcast_shape_matmul(&shape2);
        assert_eq!(result.unwrap(), (Shape::from([5, 4, 3]), Shape::from([5, 3, 4])));
    }

}
