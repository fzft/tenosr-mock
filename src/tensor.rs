use std::ops::Deref;
use std::sync::RwLock;
use std::sync::Arc;

use crate::storage::Storage;
use crate::layout::Layout;
use crate::dtype::Dtype;
use crate::device::Device;
use crate::shape::{Shape, ShapeWithOneHole};
use crate::op::BackpropOp;
use crate::cpu_backend::CpuStorage;


pub struct Tensor_ {
    storage: Arc<RwLock<Storage>>,
    layout: Layout,
    dtype: Dtype,
    device: Device,
    is_var: bool,
    op: BackpropOp,
}

#[derive(Clone)]
pub struct Tensor(Arc<Tensor_>);

impl Deref for Tensor {
    type Target = Tensor_;
    fn deref(&self) -> &Self::Target {
        &self.0.as_ref()
    }
}

impl Tensor {
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
        is_val: bool,
    ) -> Result<Self, Box<dyn std::error::Error>> {
        let shape = shape.into();
        let dtype = Dtype::F64;
        let storage = device.randn(mean, std, &shape, &dtype)?;
        let none = BackpropOp::none();
        let tensor = from_storage(storage, shape, none, is_val);
        Ok(tensor)
    }

    // Return a tensor that is a transpose of the input tensor, the two last dimensions are swapped
    pub fn t(&self) -> Result<Self, Box<dyn std::error::Error>> {
        let rank = self.rank();
        if rank < 2 {
            return Err("Invalid shape for transpose".into());
        }
        let dim1 = rank - 1;
        let dim2 = rank - 2;
        self.transpose(dim2, dim1)
    }

    pub fn transpose(&self, dim1: usize, dim2: usize) -> Result<Self, Box<dyn std::error::Error>> {
        if dim1 == dim2 {
            return Ok(self.clone());
        }

        let layout = self.layout.transpose(dim1, dim2)?;
        let storage = self.storage.clone();
        let op = BackpropOp::none();
        let tensor = Tensor_ {
            storage,
            layout,
            device: self.device,
            dtype: self.dtype,
            is_var: self.is_var,
            op,
        };
        Ok(Tensor(Arc::new(tensor)))
    }

    // self : tensor with dim b1, b2, b3 ... , bi, m, k
    // other : tensor with dim b1, b2, b3 ... , bi, k, n
    // return : tensor with dim b1, b2, b3 ... , bi, m, n
    pub fn matmul(&self, other: &Self) -> Result<Self, Box<dyn std::error::Error>> {
        let (self_shape, other_shape) = (self.layout.shape(), other.layout.shape());
        let (self_dims, other_dims) = (self_shape.dims(), other_shape.dims());
        if self_dims.len() < 2 || other_dims.len() < 2 {
            return Err("Invalid shape for matmul".into());
        }
        let (self_m, self_k) = (self_dims[self_dims.len() - 2], self_dims[self_dims.len() - 1]);
        let (other_k, other_n) = (other_dims[other_dims.len() - 2], other_dims[other_dims.len() - 1]);
        let self_b = self_dims[..self_dims.len() - 2].iter().product();
        let other_b = other_dims[..other_dims.len() - 2].iter().product();
        if self_k != other_k || self_b != other_b{
            return Err("Invalid shape for matmul".into());
        }
        let c_shape = Shape::from(&self_dims[..self_dims.len() - 2]).extend(&[self_m, other_n]);
        let storage = self.storage.read().unwrap().matmul(
            &*other.storage.read().unwrap(),
            (self_b, self_m, other_n, self_k),
            &self.layout,
            &other.layout,
        )?;

        let op = BackpropOp::none();
        Ok(from_storage(storage, c_shape, op, false))
    }

    pub fn arange(start: f64, end: f64, step: f64, dtype: Dtype, device: &Device) -> Result<Self, Box<dyn std::error::Error>> {
        let mut data = vec![];
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

    pub fn reshape<S: ShapeWithOneHole>(&self, s: S) -> Result<Self, Box<dyn std::error::Error>> {
        let shape = s.into_shape(self.layout.shape().elem_count())?;
        if shape.elem_count() != self.layout.shape().elem_count() {
            return Err("Invalid shape for reshape".into());
        }

        let op = BackpropOp::none();
        if self.layout.is_contiguous() {
            let tenosr_ = Tensor_ {
                storage: self.storage.clone(),
                layout: Layout::contiguous_with_offset(shape, self.layout.start_offset()),
                device: self.device.clone(),
                dtype: self.dtype,
                is_var: self.is_var,
                op,
            };
            Ok(Tensor(Arc::new(tenosr_)))
        } else {
            let mut storage = self.device.zeros(&shape, &self.dtype)?;
            self.storage.read().unwrap().copy_strided_src(&mut storage, 0, &self.layout)?;
            Ok(from_storage(storage, shape, op, false))
        }
    }

    pub fn affine(&self, mul: f64, add: f64) -> Result<Self, Box<dyn std::error::Error>> {
        let storage = self.storage.read().unwrap().affine(&self.layout, mul, add)?;
        let op = BackpropOp::none();
        Ok(from_storage(storage, self.layout.shape().clone(), op, false))
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
        rhs - self
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
        storage,
        layout,
        device,
        op,
        dtype,
        is_var,
    };
    Tensor(Arc::new(tensor))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ones() {
        let shape = Shape::from(vec![2, 2]);
        let device = Device::Cpu;
        let tensor = Tensor::ones(shape, Dtype::F64, &device).unwrap();
        let storage = tensor.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };

        assert_eq!(storage.device(), Device::Cpu);
        assert_eq!(storage.dtype(), Dtype::F64);

        assert_eq!(data, &[1.0, 1.0, 1.0, 1.0]);
        assert_eq!(tensor.layout.dims(), &[2, 2]);
        assert_eq!(tensor.layout.strides(), &[2, 1]);
        assert_eq!(tensor.layout.start_offset(), 0);
    }

    #[test]
    fn zeros() {
        let shape = Shape::from(vec![2, 2]);
        let device = Device::Cpu;
        let tensor = Tensor::zeros(shape, Dtype::F64, &device).unwrap();
        let storage = tensor.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };

        assert_eq!(storage.device(), Device::Cpu);
        assert_eq!(storage.dtype(), Dtype::F64);

        assert_eq!(data, &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(tensor.layout.dims(), &[2, 2]);
        assert_eq!(tensor.layout.strides(), &[2, 1]);
        assert_eq!(tensor.layout.start_offset(), 0);
    }

    #[test]
    fn rand_uniform() {
        let shape = Shape::from(vec![2, 2]);
        let device = Device::Cpu;
        let tensor = Tensor::rand_uniform(0.0, 1.0, shape, &device, false).unwrap();
        let storage = tensor.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };

        assert_eq!(storage.device(), Device::Cpu);
        assert_eq!(storage.dtype(), Dtype::F64);

        assert_eq!(tensor.layout.dims(), &[2, 2]);
        assert_eq!(tensor.layout.strides(), &[2, 1]);
        assert_eq!(tensor.layout.start_offset(), 0);
        println!("{:?}", data);
    }

    #[test]
    fn randn() {
        let shape = Shape::from(vec![2, 2]);
        let device = Device::Cpu;
        let tensor = Tensor::randn(0.0, 1.0, shape, &device, false).unwrap();
        let storage = tensor.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };

        assert_eq!(storage.device(), Device::Cpu);
        assert_eq!(storage.dtype(), Dtype::F64);

        assert_eq!(tensor.layout.dims(), &[2, 2]);
        assert_eq!(tensor.layout.strides(), &[2, 1]);
        assert_eq!(tensor.layout.start_offset(), 0);
        println!("{:?}", data);
    }

    #[test]
    fn t() {
        let device = Device::Cpu;
        let a = Tensor::ones(Shape::from(vec![2, 3]), Dtype::F64, &device).unwrap();
        assert_eq!(a.layout.dims(), &[2, 3]);
        assert_eq!(a.layout.strides(), &[3, 1]);

        let b = a.t().unwrap();
        let storage = b.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };
        assert_eq!(data, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(b.layout.dims(), &[3, 2]);
        assert_eq!(b.layout.strides(), &[1,3]);
        assert_eq!(b.layout.start_offset(), 0);
    }

    #[test]
    fn matmul() {
        let device = Device::Cpu;
        let a = Tensor::ones(Shape::from(vec![2, 10000]), Dtype::F64, &device).unwrap();
        let b = Tensor::ones(Shape::from(vec![10000, 2]), Dtype::F64, &device).unwrap();
        let c = a.matmul(&b).unwrap();
        let storage = c.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };
        assert_eq!(data, &[10000.0, 10000.0, 10000.0, 10000.0]);
        assert_eq!(c.layout.dims(), &[2, 2]);
        assert_eq!(c.layout.strides(), &[2, 1]);
        assert_eq!(c.layout.start_offset(), 0);
    }

    // Transpose matmul
    #[test]
    fn matmul_t_with_accelerate() {
        let device = Device::Cpu;
        let a = Tensor::ones(Shape::from(vec![10000, 2]), Dtype::F64, &device).unwrap();
        let b = Tensor::ones(Shape::from(vec![10000, 2]), Dtype::F64, &device).unwrap();
        let c = a.t().unwrap().matmul(&b).unwrap();
        let storage = c.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };
        assert_eq!(data, &[10000.0, 10000.0, 10000.0, 10000.0]);
        assert_eq!(c.layout.dims(), &[2, 2]);
        assert_eq!(c.layout.strides(), &[2, 1]);
        assert_eq!(c.layout.start_offset(), 0);
    }

    #[test]
    fn affine() {
        let device = Device::Cpu;
        let a = Tensor::ones(Shape::from(vec![2, 2]), Dtype::F64, &device).unwrap();
        let b = a.affine(2.0, 1.0).unwrap();
        let storage = b.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };
        assert_eq!(data, &[3.0, 3.0, 3.0, 3.0]);
        assert_eq!(b.layout.dims(), &[2, 2]);
        assert_eq!(b.layout.strides(), &[2, 1]);
        assert_eq!(b.layout.start_offset(), 0);
    }

    #[test]
    fn arange() {
        let device = Device::Cpu;
        let a = Tensor::arange(0.0, 10.0, 1.0, Dtype::F64, &device).unwrap();
        let storage = a.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };
        assert_eq!(data, &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0]);
        assert_eq!(a.layout.dims(), &[10]);
        assert_eq!(a.layout.strides(), &[1]);
        assert_eq!(a.layout.start_offset(), 0);
    }

    #[test]
    fn arange_reverse() {
        let device = Device::Cpu;
        let a = Tensor::arange(10.0, 0.0, -1.0, Dtype::F64, &device).unwrap();
        let storage = a.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };
        assert_eq!(data, &[10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0, 2.0, 1.0]);
        assert_eq!(a.layout.dims(), &[10]);
        assert_eq!(a.layout.strides(), &[1]);
        assert_eq!(a.layout.start_offset(), 0);
    }

    #[test]
    fn arange_empty() {
        let device = Device::Cpu;
        let a = Tensor::arange(0.0, 0.0, 1.0, Dtype::F64, &device).unwrap();
        let storage = a.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };
        assert_eq!(data, &[]);
        assert_eq!(a.layout.dims(), &[0]);
        assert_eq!(a.layout.strides(), &[1]);
        assert_eq!(a.layout.start_offset(), 0);
    }

    #[test]
    fn arange_step_2() {
        let device = Device::Cpu;
        let a = Tensor::arange(0.0, 10.0, 2.0, Dtype::F64, &device).unwrap();
        let storage = a.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };
        assert_eq!(data, &[0.0, 2.0, 4.0, 6.0, 8.0]);
        assert_eq!(a.layout.dims(), &[5]);
        assert_eq!(a.layout.strides(), &[1]);
        assert_eq!(a.layout.start_offset(), 0);
    }

    #[test]
    fn reshape() {
        let device = Device::Cpu;
        let a = Tensor::ones(Shape::from(vec![2, 3]), Dtype::F64, &device).unwrap();
        let b = a.reshape(Shape::from(vec![3, 2])).unwrap();
        let storage = b.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };
        assert_eq!(data, &[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]);
        assert_eq!(b.layout.dims(), &[3, 2]);
        assert_eq!(b.layout.strides(), &[2, 1]);
        assert_eq!(b.layout.start_offset(), 0);
    }

    #[test]
    #[should_panic]
    fn reshape_invalid() {
        let device = Device::Cpu;
        let a = Tensor::ones(Shape::from(vec![2, 3]), Dtype::F64, &device).unwrap();
        a.reshape(Shape::from(vec![3, 3])).unwrap();
    }

    #[test]
    fn add_scalar() {
        let device = Device::Cpu;
        let a = Tensor::ones(Shape::from(vec![2, 2]), Dtype::F64, &device).unwrap();
        let b = (a + 1.0).unwrap();
        let storage = b.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };
        assert_eq!(data, &[2.0, 2.0, 2.0, 2.0]);
        assert_eq!(b.layout.dims(), &[2, 2]);
        assert_eq!(b.layout.strides(), &[2, 1]);
        assert_eq!(b.layout.start_offset(), 0);
    }

    #[test]
    fn sub_scalar() {
        let device = Device::Cpu;
        let a = Tensor::ones(Shape::from(vec![2, 2]), Dtype::F64, &device).unwrap();
        let b = (a - 1.0).unwrap();
        let storage = b.storage.read().unwrap();
        let data = match &*storage {
            Storage::Cpu(storage) => match storage {
                CpuStorage::F64(data) => data,
                _ => panic!("Invalid storage type")
            },
            _ => panic!("Invalid device")
        };
        assert_eq!(data, &[0.0, 0.0, 0.0, 0.0]);
        assert_eq!(b.layout.dims(), &[2, 2]);
        assert_eq!(b.layout.strides(), &[2, 1]);
        assert_eq!(b.layout.start_offset(), 0);
    }

}


