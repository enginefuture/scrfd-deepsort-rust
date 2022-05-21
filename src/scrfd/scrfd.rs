use anyhow::anyhow;
use anyhow::Result;
use ndarray as np;
use ndarray::ArrayView1;
use ndarray::ArrayView2;
use ndarray::ArrayView3;
use np::s;
use opencv::{self as cv, prelude::*};

pub struct SCRFD{
    inp_width:i32,                             // 宽度
    inp_height: i32,                           // 高度
    conf_threshold: f32,                       // 临界点  0.5
    nms_threshold: f32,                        // 临界点  0.5
    net: cv::dnn::Net,                         // 模型
    keep_ratio: bool,                          // 保持比率
    fmc: usize,                                  //
    _feat_stride_fpn: Vec<i32>,         //
    _num_anchors: i32,                         // 锚
}

impl Default for SCRFD {
    fn default() -> Self {
        Self { inp_width: Default::default(), inp_height: Default::default(), conf_threshold: Default::default(), nms_threshold: Default::default(), net: cv::dnn::Net::default().unwrap(), keep_ratio: Default::default(), fmc: Default::default(), _feat_stride_fpn: Default::default(), _num_anchors: Default::default() }
    }
}


impl SCRFD{

    pub fn new(conf_threshold: f32, nms_threshold: f32, onnxmodel:&str) -> Result<Self>{
        // cv::dnn::read_net_from_onnx(onnx_file)

        let mut net = cv::dnn::read_net(onnxmodel, "", "")?;
        net.set_preferable_backend(cv::dnn::DNN_BACKEND_CUDA);
        net.set_preferable_target(cv::dnn::DNN_TARGET_CUDA);
        Ok(SCRFD{
            inp_width: 640,
            inp_height: 640,
            conf_threshold: conf_threshold,
            nms_threshold: nms_threshold,
            net: net,
            keep_ratio: true,
            fmc: 3,
            _feat_stride_fpn: [8, 16, 32].to_vec(),
            _num_anchors: 2,
        })
    }

    /**
     * 修改图片的尺寸
     */
    fn resize_image(&self, srcimg: &cv::core::Mat) -> Result<(cv::core::Mat, i32, i32, i32, i32)>{
        let (mut padh, mut padw, mut newh, mut neww) = (0,0,self.inp_height, self.inp_width);
        // 图片尺寸结构体
        let size = srcimg.size()?;
        // 图片高
        let height = size.height;
        // 图片宽
        let width = size.width;
        let mut img = cv::core::Mat::default();

        // srcimg.
        if (self.keep_ratio && height != width){

            let hw_scale = height as f32 / width as f32;
            // println!("hw_scale={:?}", hw_scale);

            if (hw_scale > 1.0){
                newh = self.inp_height;
                neww = (self.inp_width as f32/ hw_scale) as i32;
                let mut dst_img = cv::core::Mat::default();
                let size = cv::core::Size_{
                    width: neww,
                    height: newh,
                };
                cv::imgproc::resize(&srcimg, &mut dst_img, size, 0.0, 0.0, opencv::imgproc::INTER_AREA);
                padw = ((self.inp_width - neww) as f32 * 0.5) as i32;
                cv::core::copy_make_border(
                    &dst_img, 
                    &mut img, 
                    0, 
                    0, 
                    padw,
                    self.inp_height - newh - padh, 
                    cv::core::BORDER_CONSTANT, 
                    cv::core::Scalar::new(0.0, 0.0, 0.0, 0.0),
                );
            }else{
                newh = (self.inp_height as f32* hw_scale + 1.0) as i32;
                neww = self.inp_width;
                let mut dst_img = cv::core::Mat::default();
                let size = cv::core::Size_{
                    width: neww,
                    height: newh,
                };
                cv::imgproc::resize(&srcimg, &mut dst_img, size, 0.0, 0.0, opencv::imgproc::INTER_AREA);
                padh = ((self.inp_height - newh) as f32 * 0.5) as i32;
                cv::core::copy_make_border(
                    &dst_img, 
                    &mut img, 
                    padh,
                    self.inp_height - newh - padh, 
                    0, 
                    0, 
                    cv::core::BORDER_CONSTANT, 
                    cv::core::Scalar::new(0.0, 0.0, 0.0, 0.0),
                );
            }
        }else{
            let size = cv::core::Size_{
                width: self.inp_width,
                height: self.inp_height,
            };
            cv::imgproc::resize(&srcimg, &mut img, size, 0.0, 0.0, opencv::imgproc::INTER_AREA);

        }

        Ok((img, newh, neww, padh ,padw))
    }

    /**
     * 点
     */
    fn distance2bbox(&self, 
        points:np::ArrayBase<np::OwnedRepr<f32>, np::Dim<[usize; 2]>>, 
        distance:np::ArrayBase<np::OwnedRepr<f32>, np::Dim<[usize; 2]>>
    )-> Result<np::ArrayBase<np::OwnedRepr<f32>, np::Dim<[usize; 2]>>>{
        let x11 = points.slice(s![..,0]).map({|x| *x});
        let x12 =  distance.slice(s![..,0]).map({|x| *x});
        let x1 = x11 - x12;
        
        let y11 = points.slice(s![..,1]).map({|x| *x});
        let y12 =  distance.slice(s![..,1]).map({|x| *x});
        let y1 = y11 - y12;

        let x21 = points.slice(s![..,0]).map({|x| *x});
        let x22 =  distance.slice(s![..,2]).map({|x| *x});
        let x2 = x21 + x22;

        let y21 = points.slice(s![..,1]).map({|x| *x});
        let y22 =  distance.slice(s![..,3]).map({|x| *x});
        let y2 = y21 + y22;
        let mut bbox:Vec<[f32;4]> = Vec::new();
        
        for (i,_) in x1.iter().enumerate(){
            let elem= [x1[i], y1[i], x2[i], y2[i]];
            bbox.push(elem);
        }
        let bbox = np::arr2(&bbox);
        Ok(bbox)
    }


    /**
     * 点
     */
    fn distance2kps(&self, 
        points:np::ArrayBase<np::OwnedRepr<f32>, np::Dim<[usize; 2]>>, 
        distance:np::ArrayBase<np::OwnedRepr<f32>, np::Dim<[usize; 2]>>
    )-> Result<np::ArrayBase<np::OwnedRepr<f32>, np::Dim<[usize; 2]>>>{
        let mut preds = vec![];
        for i in (0..distance.shape()[1]).step_by(2){
            let px1 = points.slice(s![..,i % 2]).map({|x| *x});
            let px2 = distance.slice(s![..,i]).map({|x| *x});
            let px = px1 + px2;

            let py1 = points.slice(s![..,i % 2 + 1]).map({|x| *x});
            let py2 = distance.slice(s![..,i + 1]).map({|x| *x});
            let py = py1 + py2;
            // println!("px={:?}", px);
            
            // preds.push(axis, px);
            
            // preds.push(axis, px.slice(s!([..])));
            preds.push(px);
            preds.push(py);
        }
        let mut preds_list:Vec<[f32;10]> = Vec::new();       

        for i in 0..preds[0].shape()[0]{
            let p = [
                preds[0].get(i).unwrap().clone(),
                preds[1].get(i).unwrap().clone(),
                preds[2].get(i).unwrap().clone(),
                preds[3].get(i).unwrap().clone(),
                preds[4].get(i).unwrap().clone(),
                preds[5].get(i).unwrap().clone(),
                preds[6].get(i).unwrap().clone(),
                preds[7].get(i).unwrap().clone(),
                preds[8].get(i).unwrap().clone(),
                preds[9].get(i).unwrap().clone(),
            ];
            preds_list.push(p);
        }
        
        let preds = np::arr2(&preds_list);
        Ok(preds)
    }

    /**
     * 主
     */
    pub fn detect(&mut self, mut srcimg: cv::core::Mat) -> Result<(cv::core::Mat, Vec<[f32;4]>, Vec<f32>)>{
        let (img, newh, neww, padh, padw) = self.resize_image(&srcimg)?;

        let size = cv::core::Size_{
            width: self.inp_width,
            height: self.inp_height,
        };

        let blob = cv::dnn::blob_from_image(
            &img,
            1.0 / 128.0,
            size,
            cv::core::Scalar::new(127.5, 127.5, 127.5, 0.0),
            true,
            false,
            cv::core::CV_32F
        )?;
        // 设置网络的输入
        self.net.set_input(&blob, "", 1.0, cv::core::Scalar::default());
        // 运行前向传递以获取输出层的输出
        let mut outs:cv::core::Vector<cv::core::Vector<cv::core::Mat>> = cv::core::Vector::new();

        self.net.forward_and_retrieve(&mut outs, &self.net.get_unconnected_out_layers_names()?);

        // println!("{:?}", blob);
        // println!("{:?}", blob.mat_size());
        
        // 推理输出
        let (mut scores_list, mut bboxes_list, mut kpss_list) = (Vec::new(), Vec::new(), Vec::new());
        // mat转数组
        for (idx, stride) in self._feat_stride_fpn.iter().enumerate(){

            // let t:&[f32] = scores.data_typed()?;
            // println!("score.data:={:?}", t);
            // println!("scores:={:?}", scores);
            // println!("scores.mat_size:={:?}", scores.mat_size());
            // 转成ndarray
            let scores = outs.get(idx * self.fmc)?.get(0)?;
            let scores_array = scores.try_as_array2()?;
            let scores_array = scores_array.map(|x| *x);

            let bbox_preds = outs.get(idx * self.fmc + 1)?.get(0)?;
            let bbox_preds_array = bbox_preds.try_as_array2()?;
            let bbox_preds_array = bbox_preds_array.map(|x| *x * *stride as f32);

            let kps_preds = outs.get(idx * self.fmc + 2)?.get(0)?;
            let kps_preds_array = kps_preds.try_as_array2()?;
            let kps_preds_array = kps_preds_array.map(|x| *x * *stride as f32);

            let height = blob.mat_size()[2] / stride;
            let width = blob.mat_size()[3] / stride;
            // println!("height:={:?}", height);
            // println!("width:={:?}", width);
            // println!("scores_array={:?}", scores_array);
            // println!("bbox_preds_array={:?}", bbox_preds_array);
            // println!("kps_preds_array={:?}", kps_preds_array);


            let mut anchor_centers:Vec<[f32;2]> = Vec::new();
            for i in 0..height{
                
                for j in 0..width{
                    let elem= [(j* *stride) as f32,(i * *stride) as f32];
                    anchor_centers.push(elem);
                    
                }
            }

            let mut anchor_centers_num:Vec<[f32;2]> = Vec::new();
            if self._num_anchors > 1{              
                for item in anchor_centers{
                    for i in 0..self._num_anchors{
                        anchor_centers_num.push(item.clone());
                    }
                }
            }else{
                for item in anchor_centers{
                    anchor_centers_num.push(item.clone());
                }
            }
            let anchor_centers_num = np::arr2(&anchor_centers_num);
            // println!("anchor_centers_num={:?}", anchor_centers_num);
            
            
            // 下标
            let mut pos_inds:Vec<usize> = Vec::new();

            for (i,pos) in scores_array.iter().enumerate(){
                if *pos >= self.conf_threshold{
                    pos_inds.push(i);
                }
            }

            for i in pos_inds.clone(){
                let data = scores_array.slice(s![i,..]).map(|x| *x);
                scores_list.push(data);
            }
            // anchor_centers_num;bbox_preds_array
            // let t:np::ArrayBase<np::OwnedRepr<f32>, np::Dim<[usize; 2]>>
            let bboxes = self.distance2bbox(anchor_centers_num.clone(), bbox_preds_array)?;
            // println!("bboxes={:?}", bboxes);
            
            for i in pos_inds.clone(){
                let data = bboxes.slice(s![i,..]).map(|x| *x);
                bboxes_list.push(data);
            }

            
            // anchor_centers_num;kps_preds_array
            // let t:np::ArrayBase<np::OwnedRepr<f32>, np::Dim<[usize; 2]>>
            let kpss = self.distance2kps(anchor_centers_num.clone(), kps_preds_array.clone())?;
            // println!("kpss={:?}", kpss);
            let kpss = kpss.clone().into_shape((kpss.shape()[0], 5, 2))?;
            for i in pos_inds.clone(){
                let data = kpss.slice(s![i,..,..]).map(|x| *x);
                kpss_list.push(data);
            }

            // println!("scores_list={:?}", scores_list);
            // println!("bboxes_list={:?}", bboxes_list);
            // println!("kpss_list={:?}", kpss_list);
        }
        
        let (ratioh, ratiow) = (srcimg.mat_size()[0] as f64 / newh as f64, srcimg.mat_size()[1] as f64 / neww as f64);
        // println!("newh={:?}", newh);
        // println!("neww={:?}", neww);
        // println!("padh={:?}", padh);
        // println!("padw={:?}", padw);
        // println!("srcimg={:?}", srcimg);
        // println!("srcimg.mat_size()={:?}", srcimg.mat_size());
        // println!("ratioh={:?}", ratioh);
        // println!("ratiow={:?}", ratiow);
        // 处理scores
        let mut scores_nms:cv::core::Vector<f32> = cv::core::Vector::new();
        let mut scores:Vec<f32> = vec![];
        for item in scores_list{
            scores.push(*item.get(0).unwrap());
            scores_nms.push(*item.get(0).unwrap());
        }
        // println!("scores={:?}", scores_nms);
        // 处理bboxes
        let mut bboxes_nms:cv::core::Vector<cv::core::Rect> = cv::core::Vector::new();
        let mut bboxes:Vec<[f32;4]> = vec![];
        for item in bboxes_list{
            let mut item1 = *item.get(0).unwrap();
            let mut item2 = *item.get(1).unwrap();
            let mut item3 = *item.get(2).unwrap();
            let mut item4 = *item.get(3).unwrap();

            item3 = item3 - item1;
            item4 = item4 - item2;
            
            item1 = ((item1 - padw as f32) as f64 * ratiow) as f32;
            item2 = ((item2 - padh as f32) as f64 * ratioh) as f32;
            item3 = (item3 as f64 * ratiow) as f32;
            item4 = (item4 as f64 * ratioh) as f32;
            bboxes_nms.push(cv::core::Rect::new(item1 as i32, item2 as i32, item3 as i32, item4 as i32));
            bboxes.push([item1, item2, item3, item4]);
        }
        // println!("bboxes={:?}", bboxes_nms);
        // 处理kpss
        let mut kpss:Vec<Vec<[f32;2]>> = vec![];
        for item in kpss_list{
            // println!("item={:?}", item);
            let mut t_:Vec<[f32;2]> = vec![];
            for i in 0..item.shape()[0]{
                let t1 = ((*item.get((i, 0)).unwrap() - padw as f32) as f64 * ratiow) as f32;
                let t2 = ((*item.get((i, 1)).unwrap() - padh as f32) as f64 * ratioh) as f32;
                t_.push([t1, t2]);
            }
            kpss.push(t_);
        }
        // println!("kpss={:?}",kpss);
        let mut indices:cv::core::Vector<i32> = cv::core::Vector::new();
        cv::dnn::nms_boxes(&bboxes_nms, &scores_nms, self.conf_threshold, self.conf_threshold, &mut indices, 1.0, 0);
        // println!("indices={:?}", indices);
        let mut bboxes_out:Vec<[f32;4]> = vec![];
        let mut scores_out:Vec<f32> = vec![];
        for i in indices{
            let xmin = bboxes[i as usize][0] as i32;
            let ymin = bboxes[i as usize][1] as i32;
            let xmax = (bboxes[i as usize][0] + bboxes[i as usize][2]) as i32;
            let ymax = (bboxes[i as usize][1] + bboxes[i as usize][3]) as i32;
            bboxes_out.push(
                [bboxes[i as usize][0]
                , bboxes[i as usize][1]
                , bboxes[i as usize][0] + bboxes[i as usize][2]
                , bboxes[i as usize][1] + bboxes[i as usize][3]]
            );
            scores_out.push(scores[i as usize]);
            // let color = cv::core::Scalar::new(0.0, 0.0, 255.0);
            // *****************************画图************************************
            // 方框
            // cv::imgproc::rectangle(
            //     &mut srcimg
            //     , cv::core::Rect::from_points(cv::core::Point::new(xmin, ymin), cv::core::Point::new(xmax, ymax))
            //     , cv::core::VecN([255., 0., 0., 0.])
            //     , 2
            //     , cv::imgproc::LINE_8
            //     , 0
            // );
            // 点
            // for j in 0..5{
            //     cv::imgproc::circle(&mut srcimg
            //         , cv::core::Point::new(kpss[i as usize][j as usize][0] as i32, kpss[i as usize][j as usize][1] as i32)
            //         , 1
            //         , cv::core::VecN([0., 255., 0., 0.])
            //         , -1
            //         , cv::imgproc::LINE_8
            //         , 0
            //     );
            // }
            // 文本 ,准确度
            // let t = &scores[i as usize].to_string()[..] ;
            // cv::imgproc::put_text(&mut srcimg
            //     , &scores[i as usize].to_string()[..]
            //     , cv::core::Point::new(xmin, ymin - 10)
            //     , cv::imgproc::FONT_HERSHEY_SIMPLEX
            //     , 1.0
            //     , cv::core::VecN([0., 255., 0., 0.])
            //     , 1
            //     , cv::imgproc::LINE_8
            //     , false
            // );
            // *****************************画图************************************

        }
        Ok((srcimg, bboxes_out, scores_out))
    }
}

trait AsArray{
    fn try_as_array1(&self) -> Result<ArrayView1<f32>>;
    fn try_as_array2(&self) -> Result<ArrayView2<f32>>;
    fn try_as_array3(&self) -> Result<ArrayView3<f32>>;
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
        let bytes:&[f32] = self.data_typed()?;
        let size = self.size()?;
        let dim = self.mat_size()[2] as usize;
        let a = ArrayView2::from_shape((size.width as usize, dim), bytes)?;
        Ok(a)
    }

    fn try_as_array3(&self) -> Result<ArrayView3<f32>>{
        if !self.is_continuous(){
            return Err(anyhow!("Mat is not continuous"));
        }
        // 提取数据
        let bytes:&[f32] = self.data_typed()?;
        let size = self.size()?;
        let dim = self.mat_size()[2] as usize;
        let a = ArrayView3::from_shape((size.height as usize,size.width as usize, dim), bytes)?;
        Ok(a)
    }
}


#[cfg(test)]
mod tests{
    use super::*;


    // 单张图片人脸检测
    #[test]
    fn test()-> Result<()>{
        let mut mynet = SCRFD::new(0.5, 0.5, "./src/onnx/scrfd_500m_kps.onnx")?;
        let mut srcimg = cv::imgcodecs::imread("./src/s_l.jpg", cv::imgcodecs::IMREAD_COLOR)?;
        let (outimg,_ ,_) = mynet.detect(srcimg)?;
        let win_name = "Deep learning object detection in OpenCV";
        cv::highgui::named_window(win_name, 0);
        cv::highgui::imshow(win_name, &outimg);
        cv::highgui::wait_key(0);
        cv::highgui::destroy_all_windows();
        Ok(())
    }

    // 视频流人脸检测
    #[test]
    fn test_stream()-> Result<()>{
        let mut mynet = SCRFD::new(0.5, 0.5, "./src/onnx/scrfd_500m_kps.onnx")?;

        let window = "video capture";
        cv::highgui::named_window(window, 1)?;
        #[cfg(feature = "opencv-32")]
        let mut cam = cv::videoio::VideoCapture::new_default(0)?;  // 0 is the default camera
        #[cfg(not(feature = "opencv-32"))]
        // let params = cv::core::Vector::default();
        // let mut cam = cv::videoio::VideoCapture::from_file_with_params("rtmp://127.0.0.1/myapp/test", cv::videoio::CAP_ANY, &params)?;
        let mut cam = cv::videoio::VideoCapture::new(1, cv::videoio::CAP_ANY)?;  // 0 is the default camera
        let opened = cv::videoio::VideoCapture::is_opened(&cam)?;
        if !opened {
            panic!("Unable to open default camera!");
        }
        loop {
            let mut frame = cv::core::Mat::default();
            cam.read(&mut frame)?;
            let (mut frame, _, _) = mynet.detect(frame)?;

            if frame.size()?.width > 0 {
                cv::highgui::imshow(window, &mut frame)?;
            }
            let key = cv::highgui::wait_key(10)?;
            if key > 0 && key != 255 {
                break;
            }
        }
        Ok(())
    }


    // cv相机测试
    #[test]
    fn test_photo()-> Result<()>{

        let window = "video capture";
        cv::highgui::named_window(window, 1)?;
        #[cfg(feature = "opencv-32")]
        let mut cam = cv::videoio::VideoCapture::new_default(0)?;  // 0 is the default camera
        #[cfg(not(feature = "opencv-32"))]
        let mut cam = cv::videoio::VideoCapture::new(0, cv::videoio::CAP_ANY)?;  // 0 is the default camera
        // let opened = cv::videoio::VideoCapture::is_opened(&cam)?;
        // if !opened {
        //     panic!("Unable to open default camera!");
        // }
        loop {
            let mut frame = cv::core::Mat::default();
            println!("2");
            cam.read(&mut frame)?;
            println!("1");
            if frame.size()?.width > 0 {
                cv::highgui::imshow(window, &mut frame)?;
            }
            let key = cv::highgui::wait_key(10)?;
            if key > 0 && key != 255 {
                break;
            }
        }
        Ok(())
    }
    
}