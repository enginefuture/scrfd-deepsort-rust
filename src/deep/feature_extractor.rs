use anyhow::anyhow;
use anyhow::Result;
use chrono::DateTime;
use chrono::Local;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayView3;
use ndarray::s;
use ndarray as np;
use tch::nn::Module;
use tch::CModule;
use crate::deep::model::Net;
use opencv::{self as cv, prelude::*};
use tch::{nn, Tensor, Device};


#[derive(Debug)]
pub struct Extractor{
    net: Net,                // torch 网络
    net_path: String,        // 网络路径
    state: CModule,
    device: Device,          // 设备
    size: (i32, i32),

}


impl Extractor{
    pub fn new(model_path: &str, use_cuda: bool) -> Extractor{
        let mut extractor = Extractor::default();

        extractor.device = Device::cuda_if_available();
        let mut vs = nn::VarStore::new(extractor.device);
        extractor.net = Net::new(vs.root(), 512, true);
        extractor.net_path = "./src/deep/checkpoint/deep_model.pt".to_string();
        extractor.state = tch::jit::CModule::load("./src/deep/checkpoint/deep_model.pt".to_string()).unwrap();

        extractor.size = (64, 128);


        extractor
    }

    pub fn _preprocess(&mut self, im_crops: Vec<cv::core::Mat>) -> Result<Tensor>{

        fn _resize(im: cv::core::Mat, size: (i32, i32)) -> Result<Tensor>{

            // println!("{:?}", im);
            // // 数据类型转换
            // let mut im_f32 = cv::core::Mat::default();
            // im.convert_to(&mut im_f32, cv::core::CV_32FC3, 1.0, 0.0);
            
            // println!("{:?}", im_f32.elem_size());
            // println!("{:?}", im_f32.elem_size1());
            // let t = im_f32.data_typed::<cv::core::Vec3f>()?;
            // println!("{:?}", t);
            // im_f32.try_div_op(255.0);
            // let t = im_f32.try_as_array3()?;
            // println!("{:?}", t);
            // let mut im = im.map(|x| *x as f32 / 255.);
            // cv::imgcodecs::imreadmulti(filename, mats, flags)
            
            // println!("{:?}", im);
            let mut dst = cv::core::Mat::default();
            
            cv::imgproc::resize(
                &im
                , &mut dst,
                 cv::core::Size::new(size.0, size.1)
                 , 0.0
                 , 0.0
                 , cv::imgproc::INTER_LINEAR
                );
            // println!("dst={:?}", dst);
            
            let mut out = dst.try_as_array3()?.map(|x| *x as f32 / 255.0);
            // ndarray转tensor
            let mut td = vec![];
            for item in out.rows().clone(){
                let x = (item[0] - 0.485) / 0.229;
                td.push(x);
            }
            for item in out.rows().clone(){
                let x = (item[1] - 0.456) / 0.224;
                td.push(x);
            }
            for item in out.rows().clone(){
                let x = (item[2] - 0.406) / 0.225;
                td.push(x);
            }
            
            let tensor = Tensor::f_of_slice(&td)?;
            let tensor = tensor.reshape(&[3, 128, 64]);
            let tensor = tensor.unsqueeze(0);

            // println!("{:?}", tensor);
            Ok(tensor)
        }
        let mut tensors = vec![];
        for im in im_crops{
            let tensor = _resize(im, self.size)?;
            tensors.push(tensor);
        }

        let im_batch = tch::Tensor::cat(&tensors, 0);

        Ok(im_batch)

    }


    pub fn call(&mut self, im_crops: Vec<cv::core::Mat>) -> Result<Tensor>{
        
        let im_batch = self._preprocess(im_crops)?;
        let mut features = Tensor::default();
        tch::no_grad(||{
            let im_batch = im_batch.to(self.device);
            features = self.state.forward(&im_batch);
        });
        Ok(features)
        // let mut features_vec = vec![];

        // for i in 0..features.size()[0]{
        //     let mut sub_features = vec![];
        //     for j in 0..features.size()[1]{
        //         let e = features.double_value(&[i,j]);
        //         sub_features.push(e);
        //     }
        //     features_vec.push(sub_features);
        // }
        // // println!("features_vec={:?}",features.print());
        // Ok(features_vec)       
    }


}

impl Default for Extractor{
    fn default() -> Self {
        Self { 
            net: Default::default()
            , device: Device::cuda_if_available()
            , size: Default::default()
            , net_path: Default::default(),
            state: tch::jit::CModule::load("./src/deep/checkpoint/deep_model.pt".to_string()).unwrap(),
        }
    }
}


trait AsArray{
    fn try_as_array3(&self) -> Result<ArrayView3<u8>>;
    fn try_div_op(&mut self, op:f32) -> Result<()>;

}

impl AsArray for cv::core::Mat{


    fn try_as_array3(&self) -> Result<ArrayView3<u8>>{
        if !self.is_continuous(){
            return Err(anyhow!("Mat is not continuous"));
        }
        // 提取数据
        let bytes = self.data_bytes()?;
        
        let size = self.size()?;
        let dim = self.elem_size()?;
        let a = ArrayView3::from_shape((size.height as usize,size.width as usize, dim), bytes)?;
        Ok(a)
    }

    fn try_div_op(&mut self, op:f32) -> Result<()> {
        if !self.is_continuous(){
            return Err(anyhow!("Mat is not continuous"));
        }
        for item in self.data_typed_mut::<f32>(){
            for i in 0..item.len(){
                item[i] = item[i] / op;
                // println!("{:?}", item[1]);
            }
        }
        Ok(())
    }
}



#[cfg(test)]
mod tests{
    use tch::nn::Module;

    use super::*;

    #[test]
    fn test() ->Result<()>{
        // let t = tch::Cuda::cudnn_is_available();
        let device = Device::cuda_if_available();
        // println!("{:?}", t);

        let state = tch::jit::CModule::load("./src/deep/checkpoint/deep_model.pt")?;
        let x = Tensor::randn(&[4, 3, 128, 64], (tch::Kind::Float, device));
        let y = state.forward(&x);

        println!("{:?}", y);
        println!("{:?}", y.print());
        // let mut s = tch::nn::VarStore::new(t1);
        // tch::nn::VarStore::load(&mut s, "./src/deep/checkpoint/deep_model.pt");
        // // s.unfreeze();
        // println!("{:?}", s.variables());
        // s.trainable_variables()
        // println!("{:?}", t1.);
        Ok(())
    }

    #[test]
    fn test_t() -> Result<()>{
        let device = Device::cuda_if_available();
        let t = Tensor::new();
        let x = Tensor::randn(&[3, 128, 64], (tch::Kind::Float, device));
        // for x in x.data_ptr(){

        // }
        let mut t = vec![];
        for i in 0..10{
            t.push(i);
        }
        // println!("{:?}", t);
        let d = [1., 2., 3., 4., 5., 6., 7., 8., 9., 10., 11., 12., 13., 14., 15., 16., 17., 18.];
        let t = Tensor::f_of_slice(&d)?;
        let t1 = t.reshape(&[3, 3, 2]);
        let mut dst:Vec<f64> = vec![];
        let t2 = t1.double_value(&[0,0,0]);
        println!("{:?}", t1.size());
        println!("{:?}", t2);
        // println!("{:?}", t1.data_ptr());
        // println!("{:?}", t1.stride());
        // println!("{:?}", t1.);
        // for i in t1.iter(){

        // }
        // let t = Tensor::f_of_data_size(data, size, kind);
        // println!("{:?}", x);
        // let t = x.;
        Ok(())
    }

    // mat操作
    #[test]
    fn test_v_r_m() -> Result<()>{
        // let mut v = cv::core::Vector::new();
        // v.push(3);
        // v.push(3);
        // v.push(1);
        // v.push(4);
        // let t = unsafe{ cv::core::Mat::new_nd_vec(&v, cv::core::CV_8U)}?;
        // println!("{:?}", t);
        // let mut t = unsafe{ cv::core::Mat::new_rows_cols_with_default(328, 400, cv::core::CV_8U, cv::core::Scalar::new(255.0, 255.0, 255.0, 0.0))}?;
        
        // println!("{:?}", t);
        // let d = t.data_bytes()?;
        // println!("{:?}", d.len());

        // let mut srcimg = cv::imgcodecs::imread("./src/s_l.jpg", cv::imgcodecs::IMREAD_COLOR)?;
        // println!("{:?}", srcimg);
        // let row_range = cv::core::Range::new(10, 20)?;
        // let col_range = cv::core::Range::new(10, 20)?;

        // let t = cv::core::Mat::rowscols(&srcimg, &row_range, &col_range)?;
        // println!("{:?}", t);
        println!("{:?}", 5 & 7);

        Ok(())
    }
}