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
}