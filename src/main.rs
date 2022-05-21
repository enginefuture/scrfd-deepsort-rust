mod scrfd;
mod deep_sort;
mod deep;
mod sort;
mod test;

use anyhow::Result;
use opencv::{self as cv, prelude::*};
use scrfd::face_detect::ScrfdDeepsort;
use chrono::prelude::*;

fn draw(frame_list: Vec<[i32;5]>, frame: &mut cv::core::Mat){
    for image_info in frame_list{
        let cur_id = image_info[4];
        cv::imgproc::put_text(frame
            , &cur_id.to_string()
            , cv::core::Point::new(image_info[0], image_info[1] - 10)
            , cv::imgproc::FONT_HERSHEY_SIMPLEX
            , 1.0
            , cv::core::VecN([0., 255., 0., 0.])
            , 1
            , cv::imgproc::LINE_8
            , false
        );
        cv::imgproc::rectangle(
            frame
            , cv::core::Rect::from_points(cv::core::Point::new(image_info[0], image_info[1]), cv::core::Point::new(image_info[2], image_info[3]))
            , cv::core::VecN([255., 0., 0., 0.])
            , 2
            , cv::imgproc::LINE_8
            , 0
        );
    }
}

fn main() -> Result<()>{
    let mut yolo_reid = ScrfdDeepsort::new()?;

    let window = "video capture";
    cv::highgui::named_window(window, 1)?;
    #[cfg(feature = "opencv-32")]
    let mut cam = cv::videoio::VideoCapture::new_default(0)?;  // 0 is the default camera
    #[cfg(not(feature = "opencv-32"))]
    // let params = cv::core::Vector::default();
    // let mut cam = cv::videoio::VideoCapture::from_file_with_params("rtmp://127.0.0.1/myapp/test", cv::videoio::CAP_ANY, &params)?;
    // let mut cam = cv::videoio::VideoCapture::new(1, cv::videoio::CAP_ANY)?;  // 0 is the default camera
    let mut cam = cv::videoio::VideoCapture::from_file("/home/tab/src/rustsrc/deepsort/src/1.flv", cv::videoio::CAP_ANY)?;

    let opened = cv::videoio::VideoCapture::is_opened(&cam)?;
    if !opened {
        panic!("Unable to open default camera!");
    }
    let mut i  = 0;
    let mut sum = 0;
    loop {
        i += 1;
        let mut frame = cv::core::Mat::default();
        cam.read(&mut frame)?;
        // let mut frame = cv::imgcodecs::imread("/home/tab/src/rustsrc/deepsort/src/s_l.jpg", cv::imgcodecs::IMREAD_COLOR)?;
        // if i < 710{
        //     continue;
        // }
        // if i > 770{
        //     break;
        // }
        let start: DateTime<Local> = Local::now();
        let m1: i64 = start.timestamp_millis();

        let frame_list = yolo_reid.detect(frame.clone())?;
        println!("kkkkkkkkkkkkkkkkkk:::::{:?}::::kkkkkkkkkkkkkkkkkkkkkkk::{:?}", i, frame_list);
        let end: DateTime<Local> = Local::now();
        let m2: i64 = end.timestamp_millis();
        println!("{:?}  - {:?} = {:?}", m1, m2, m2 - m1);
        sum += m2 - m1;

        draw(frame_list, &mut frame);
        
        if frame.size()?.width > 0 {
            cv::highgui::imshow(window, &mut frame)?;
        }
        let key = cv::highgui::wait_key(10)?;
        if key > 0 && key != 255 {
            break;
        }

    }
    println!("sum={:?}", sum);
    println!("deep sum={:?}", yolo_reid.sum);

    Ok(())
}
