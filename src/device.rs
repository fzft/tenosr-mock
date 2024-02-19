use std::error::Error;
use crate::shape::Shape;
use crate::dtype::Dtype;
use crate::storage::Storage;
use crate::cpu_backend::CpuStorage;

#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Hash)]
pub enum Device {
    Cpu,
    Gpu,
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
        CpuStorage::F64(vec![*self])
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

impl<const N: usize, const M: usize > NdArray for &[[f64; M]; N] {
    fn shape(&self) -> Result<Shape, Box<dyn Error>> {
        Ok(Shape::from((N, M)))
    }

    fn to_cpu_storage(&self) -> CpuStorage {
        CpuStorage::F64(self.iter().flatten().copied().collect())
    }
}

impl<const N1: usize, const N2: usize, const N3: usize> NdArray for &[[[f64; N3]; N2]; N1] {
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

    pub fn ones(&self, shape: &Shape, dtype: &Dtype) -> Result<Storage, Box<dyn Error>> {
        match self {
            Device::Cpu => {
                let storage = CpuStorage::ones(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            _ => unimplemented!("Not implemented yet")
        }
    }

    pub fn zeros(&self, shape: &Shape, dtype: &Dtype) -> Result<Storage, Box<dyn Error>> {
        match self {
            Device::Cpu => {
                let storage = CpuStorage::zeros(shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            _ => unimplemented!("Not implemented yet")
        }
    }

    pub fn rand_uniform(&self,lo: f64, hi:f64, shape: &Shape, dtype: &Dtype) -> Result<Storage, Box<dyn Error>> {
        match self {
            Device::Cpu => {
                let storage = CpuStorage::rand_uniform(lo, hi,  shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            _ => unimplemented!("Not implemented yet")
        }
    }

    pub fn randn(&self, mean: f64, std: f64, shape: &Shape, dtype: &Dtype) -> Result<Storage, Box<dyn Error>> {
        match self {
            Device::Cpu => {
                let storage = CpuStorage::randn(mean, std, shape, dtype)?;
                Ok(Storage::Cpu(storage))
            }
            _ => unimplemented!("Not implemented yet")
        }
    }

    pub fn storage_owned(&self, data: Vec<f64>) -> Result<Storage, Box<dyn Error>> {
        match self {
            Device::Cpu => {
                let storage = CpuStorage::from(data)?;
                Ok(Storage::Cpu(storage))
            }
            _ => unimplemented!("Not implemented yet")
        }
    }

    pub fn storage<A: NdArray>(&self, data: A) -> Result<Storage, Box<dyn Error>> {
        match self {
            Device::Cpu => {
                Ok(Storage::Cpu(data.to_cpu_storage()))
            }
            _ => unimplemented!("Not implemented yet")
        }
    }
}