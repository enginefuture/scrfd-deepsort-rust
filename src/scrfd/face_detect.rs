use anyhow::Result;
use chrono::{DateTime, Local};
use crate::deep_sort::deep_sort::DeepSort;
use crate::scrfd::scrfd::SCRFD;
use opencv::{self as cv, prelude::*};


pub struct ScrfdDeepsort{
    face_detect: SCRFD,
    deepsort: DeepSort,
    pub sum:i64,
}
impl Default for ScrfdDeepsort {
    fn default() -> Self {
        Self { face_detect: Default::default(), deepsort: Default::default(), sum: 0 }
    }
}

impl ScrfdDeepsort {

    pub fn new() -> Result<Self>{
        let mut sd = ScrfdDeepsort::default();
        sd.face_detect = SCRFD::new(0.5, 0.5, "./src/onnx/scrfd_500m_kps.onnx")?;
        sd.deepsort = DeepSort::new("./src/deep/checkpoint/deep_model.pt", 0.2, 0.3, 0.5, 0.7, 70, 3, 100, true);

        Ok(sd)
    }


    pub fn detect(&mut self, mut srcimg: cv::core::Mat) -> Result<Vec<[i32; 5]>>{
        // scrfd获取人脸框，和预测率
        let (_,bboxs_xywh ,cls_conf) = self.face_detect.detect(srcimg.clone())?;
        println!("{:?}", cls_conf);
        let mut bbox_xywh = vec![];
        for bbox in bboxs_xywh{
            let x = bbox[0];
            let y = bbox[1];
            let x2 = bbox[2];
            let y2 = bbox[3];
            let bbox_left = if x < x2{x}else{x2};
            let bbox_top = if y < y2{y}else{y2};
            let bbox_w = (x2 - x).abs();
            let bbox_h = (y2 - y).abs();

            let x_c = bbox_left + bbox_w / 2.0;
            let y_c = bbox_top + bbox_h / 2.0;

            let w = bbox_w;
            let h = bbox_h;

            bbox_xywh.push([x_c, y_c, w, h])
        }
        let start: DateTime<Local> = Local::now();
        let m1: i64 = start.timestamp_millis();
        
        let outputs = self.deepsort.update(bbox_xywh, cls_conf, srcimg);
        
        let end: DateTime<Local> = Local::now();
        let m2: i64 = end.timestamp_millis();
        println!("deep_sort time:{:?}", m2 - m1);
        self.sum += m2 - m1;
        outputs
    }
}

#[cfg(test)]
mod tests{


    use super::*;



    #[test]
    fn test() -> Result<()>{
        let mut sd = ScrfdDeepsort::new()?;
        let mut srcimg = cv::imgcodecs::imread("src.//s_l.jpg", cv::imgcodecs::IMREAD_COLOR)?;
        // println!("{:?}", srcimg);
        // let t = srcimg.data_bytes()?;
        // println!("{:?}", t.len());
        // let outputs= ScrfdDeepsort::detect(srcimg)?;
        let (_,bboxs_xywh ,cls_conf) = sd.face_detect.detect(srcimg.clone())?;

        // println!("{:?}", bboxs_xywh);
        // println!("{:?}", cls_conf);

        let outputs = sd.deepsort.update(bboxs_xywh, cls_conf, srcimg);      
        Ok(())
    }
}