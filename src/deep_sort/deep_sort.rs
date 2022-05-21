use std::rc::Rc;

use anyhow::anyhow;
use anyhow::Result;
use chrono::DateTime;
use chrono::Local;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayView3;
use ndarray::s;
use ndarray as np;
use tch::{nn, Tensor, Device};
use crate::sort::detection::Detection;
use crate::sort::preprocessing::non_max_suppression;
use crate::{
    deep::feature_extractor::Extractor
    , sort::{
        tracker::Tracker
        , nn_matching::NearestNeighborDistanceMetric
    }
};
use opencv::{self as cv, prelude::*};


pub struct DeepSort{
    min_confidence: f64,
    nms_max_overlap:f64,
    extractor:Extractor,   // 未写完
    tracker:Tracker,       // 正在写
    height: usize,
    width: usize,
}


impl Default for DeepSort{
    fn default() -> Self {
        Self { min_confidence: Default::default(), nms_max_overlap: Default::default(), extractor: Default::default(), tracker: Default::default(), height: Default::default(), width: Default::default() }
    }
}

impl DeepSort{
    pub fn new(
        model_path: &str
        , max_dist: f64
        , min_confidence: f64
        , nms_max_overlap: f64
        , max_iou_distance: f64
        , max_age: i32
        , n_init: i32
        , nn_budget: i32
        , use_cuda: bool
    ) -> Self{
        let mut deep_sort = DeepSort::default();
        deep_sort.min_confidence = min_confidence;
        deep_sort.nms_max_overlap = nms_max_overlap;
        
        // 类内部实现未写完
        deep_sort.extractor = Extractor::new(model_path, use_cuda);  

        let max_cosine_distance = max_dist;

        let nn_budget = 100;

        // 类内部实现未写完
        let metric = NearestNeighborDistanceMetric::new("cosine", max_cosine_distance, nn_budget);

        // 类内部实现未写完
        deep_sort.tracker = Tracker::new(metric, max_iou_distance, max_age, n_init);

        deep_sort
    }


    pub fn update(&mut self, bbox_xywh:Vec<[f32;4]>, confidences: Vec<f32>, ori_img: cv::core::Mat) -> Result<Vec<[i32;5]>>{
        let mut outputs = vec![];
        if confidences.len() <= 0{
            
            // 大于零返回输出结果
            return Ok(outputs);
        }else{
            self.height = ori_img.mat_size()[0] as usize;
            self.width = ori_img.mat_size()[1] as usize;
            
            let start: DateTime<Local> = Local::now();
            let m1: i64 = start.timestamp_millis();

            let features:Tensor = self._get_features(bbox_xywh.clone(), ori_img)?;

            let end: DateTime<Local> = Local::now();
            let m2: i64 = end.timestamp_millis();
            println!("ttttttttttttttttttttttttttttttt::::{:?}", m2 - m1);
            let bbox_tlwh = self._xywh_to_tlwh(bbox_xywh.clone());
            println!("预测框:::{:?}", bbox_tlwh);
            
            let mut detections = vec![];
            
            for i in 0..confidences.len(){
                let conf = confidences[i] as f64;
                if conf > self.min_confidence{
                    let t = Rc::new(features.get(i.try_into().unwrap()));
                    let detection = Detection::new(bbox_tlwh[i], conf, t);
                    detections.push(detection);
                }
            }

            let mut boxes = vec![];
            let mut scores = vec![];
            for d in detections.clone(){
                boxes.push(d.tlwh);
                scores.push(d.confidence);
            }
            let indices = non_max_suppression(boxes, self.nms_max_overlap.clone(), scores);
            let mut detections_mut = vec![];
            for i in indices{
                detections_mut.push(detections[i].clone());
            }

            // 更新追踪
            self.tracker.predict();
            println!("dddd:::{:?}", self.tracker.tracks.len());
            // for track in &self.tracker.tracks{
            //     println!("dddddddddddddddddddd::{:?}", track.track_id);
            // }
            self.tracker.update(detections_mut);
            
            for track in self.tracker.tracks.clone(){
                if !track.is_confirmed() || track.time_since_update > 1{

                    continue
                }
                let bbox = track.to_tlwh();
                let (x1, y1, x2, y2) = self._tlwh_to_xyxy(bbox);
                let track_id = track.track_id;
                outputs.push([x1, y1, x2, y2, track_id]);

            }
            // if outputs.len() > 0{
            //     // 
            // }

            Ok(outputs)
        }
        
    }

    pub fn max<T:std::cmp::PartialOrd>(&self, x1: T, x2: T) -> T{
        if x1 > x2{
            x1
        }else{
            x2
        }
    }

    pub fn min<T:std::cmp::PartialOrd>(&self, x1: T, x2: T) -> T{
        if x1 < x2{
            x1
        }else{
            x2
        }
    }

    pub fn _xywh_to_tlwh(&self, bbox_xywh: Vec<[f32;4]>) -> Vec<[f64;4]>{
        let mut bbox_tlwh:Vec<[f64;4]> = vec![];
        
        for i in 0..bbox_xywh.len(){
            let e1 = [
                (bbox_xywh[i][0] - bbox_xywh[i][2] / 2.0) as f64,
                (bbox_xywh[i][1] - bbox_xywh[i][3] / 2.0) as f64,
                bbox_xywh[i][2] as f64,
                bbox_xywh[i][3] as f64,
            ];
            bbox_tlwh.push(e1);
        }
        bbox_tlwh
    }

    pub fn _xywh_to_xyxy(&self, bbox_xywh: [f32;4]) -> (i32, i32, i32, i32){
        let x = bbox_xywh[0];
        let y = bbox_xywh[1];
        let w = bbox_xywh[2];
        let h = bbox_xywh[3];
        
        // let x1 = self.max((x - w / 2.0) as i32, 0);
        // let x2 = self.min((x + w / 2.0) as i32, self.width as i32 - 1);
        // let y1 = self.max((y - h / 2.0) as i32, 0);
        // let y2 = self.min((y + h / 2.0) as i32, self.height as i32 - 1);
        let x1 = self.max(x as i32, 0);
        let x2 = self.min((x + w) as i32, self.width as i32 - 1);
        let y1 = self.max((y ) as i32, 0);
        let y2 = self.min((y + h) as i32, self.height as i32 - 1);
       
        (x1, y1, x2, y2)
    }

    /**
     * 将bbox从xtl_ytl_w_h 转换为 xc_yc_w_h
     */
    pub fn _tlwh_to_xyxy(&self, bbox_tlwh: Vec<f64>)->(i32, i32, i32, i32){
        let x = bbox_tlwh[0];
        let y = bbox_tlwh[1];
        let w = bbox_tlwh[2];
        let h = bbox_tlwh[3];

        let x1 = self.max(x, 0.0) as i32;
        let x2 = self.min(x + w, (self.width - 1) as f64) as i32;
        let y1 = self.max(y, 0.0) as i32;
        let y2 = self.min(y + h, (self.height - 1) as f64) as i32;
        (x1, y1, x2, y2)
    }

    pub fn _get_features(&mut self, bbox_xywh:Vec<[f32;4]>, ori_img: cv::core::Mat) ->Result<Tensor>{
        // let mut im_crops:Vec<np::ArrayBase<np::OwnedRepr<u8>, np::Dim<[usize; 3]>>> = vec![];
        let mut im_mat_crops:Vec<cv::core::Mat> = vec![];

        for bbox in bbox_xywh{
            let (x1, y1, x2, y2) = self._xywh_to_xyxy(bbox);
            
            let t_img = cv::core::Mat::rowscols(
                &ori_img
                , &cv::core::Range::new(y1, y2)?
                , &cv::core::Range::new(x1, x2)?
            )?.clone();
            im_mat_crops.push(t_img);
        }
        // let mut features:Vec<Vec<f64>> = vec![];
        let mut features = Tensor::default();
        
        if im_mat_crops.len() > 0{
            features = self.extractor.call(im_mat_crops)?;
        }
        Ok(features)
    }
}


trait AsArray{
    fn try_as_array1(&self) -> Result<ArrayView1<f32>>;
    fn try_as_array2(&self) -> Result<ArrayView2<f32>>;
    fn try_as_array3(&self) -> Result<ArrayView3<u8>>;
    fn try_op_sub(&mut self, op:f32) -> Result<()>;
    fn try_op_add(&mut self, op:f32) -> Result<()>;
    fn try_op_div(&mut self, op:f32) -> Result<()>;
}

impl AsArray for cv::core::Mat{

    fn try_as_array1(&self) -> Result<ArrayView1<f32>> {
        if !self.is_continuous(){
            return Err(anyhow!("Mat is not continuous"));
        }
        // 提取数据
        let bytes:&[f32] = self.data_typed()?;
        let size = self.size()?;

        let a = ArrayView1::from_shape(size.width as usize, bytes)?;
        Ok(a)
    }

    fn try_as_array2(&self) -> Result<ArrayView2<f32>> {
        if !self.is_continuous(){
            return Err(anyhow!("Mat is not continuous"));
        }
        // 提取数据
        let bytes = self.data_typed::<f32>()?;
        let size = self.size()?;
        let dim = self.mat_size()[1] as usize;
        let a = ArrayView2::from_shape((size.height as usize, dim), bytes)?;
        Ok(a)
    }

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

    fn try_op_sub(&mut self, op:f32) -> Result<()> {
        if !self.is_continuous(){
            return Err(anyhow!("Mat is not continuous"));
        }
        for item in self.data_typed_mut::<f32>(){
            for i in 0..item.len(){
                item[i] = op - item[i];
            }
        }
        Ok(())
    }

    fn try_op_add(&mut self, op:f32) -> Result<()> {
        if !self.is_continuous(){
            return Err(anyhow!("Mat is not continuous"));
        }
        for item in self.data_typed_mut::<f32>(){
            for i in 0..item.len(){
                item[i] = op + item[i];
            }
        }
        Ok(())
    }

    fn try_op_div(&mut self, op:f32) -> Result<()> {
        if !self.is_continuous(){
            return Err(anyhow!("Mat is not continuous"));
        }
        for item in self.data_typed_mut::<f32>(){
            for i in 0..item.len(){
                item[i] = op / item[i];
            }
        }
        Ok(())
    }
}



#[cfg(test)]
mod tests{

    #[test]
    fn test(){
        
    }
}