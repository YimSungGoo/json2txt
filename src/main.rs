use serde::{Deserialize, Serialize};
use std::{fs::File, io::Read, io::Write, path::PathBuf};

#[derive(Deserialize, Serialize, Debug)]
struct JsonData {
    images: Vec<Image>,
}

#[derive(Deserialize, Serialize, Debug)]
struct Image {
    annotations: Vec<Annotation>,
    file_name: String,
    image_id: String,
    width: i32,
    height: i32,
}

#[derive(Deserialize, Serialize, Debug, Clone)]
struct Annotation {
    keypoints: Option<Vec<f32>>,
    bbox: Option<Vec<f32>>,
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut file = File::open("/media/qnap/pose/OCHuman/ochuman.json").unwrap();
    let mut contents = String::new();
    file.read_to_string(&mut contents)?;
    let json_value: JsonData = serde_json::from_str(&contents).unwrap();

    for images in &json_value.images {
        let file_name = &images.file_name;
        let image_id = &images.image_id;
        let width = &images.width;
        let height = &images.height;
        let file_extension = ".txt";

        // txt파일 경로 설정
        let full_file_name = format!("{}{}", image_id, file_extension);
        let mut file_path = PathBuf::from("./labels/");
        file_path.push(full_file_name);
        let mut file = File::create(file_path)?;

        for annotation in images.annotations.clone() {
            let bbox = annotation.bbox;
            let keypoint = annotation.keypoints;
            let mut keypoints: Vec<f32> = Vec::new();

            // 데이터셋에는 keypoint가 None인경우에도 bbox좌표는 존재했는데
            // 학습시 필요없으니 None이 아닌경우에만 Bbox,keypoint를 추출하도록 합니다.
            if keypoint.is_some() {
                // OCHuman의 Bbox는 x0y0x1y1입니다.(cx0cy0wh가 아님)
                let bbox_x1 = (bbox.clone().unwrap()[0]
                    + (bbox.clone().unwrap()[2] - bbox.clone().unwrap()[0]) / 2.0)
                    / (*width) as f32;
                let bbox_y1 = (bbox.clone().unwrap()[1]
                    + (bbox.clone().unwrap()[3] - bbox.clone().unwrap()[1]) / 2.0)
                    / (*height) as f32;
                let bbox_x2 =
                    (bbox.clone().unwrap()[2] - bbox.clone().unwrap()[0]) / (*width) as f32;
                let bbox_y2 =
                    (bbox.clone().unwrap()[3] - bbox.clone().unwrap()[1]) / (*height) as f32;

                // Bbox,keypoint 좌표를 width, height로 나눈 값으로 스케일링
                // i = 36~41; head,neck (x,y,v)좌표
                let mut transformed_keypoints: Vec<f32> = Vec::new();
                for i in 0..keypoint.clone().unwrap().len() {
                    if i % 3 == 0 && i != 36 && i != 37 && i != 38 && i != 39 && i != 40 && i != 41
                    {
                        let x = keypoint.clone().unwrap()[i] / *width as f32;
                        let y = keypoint.clone().unwrap()[i + 1] / *height as f32;
                        transformed_keypoints.push(x);
                        transformed_keypoints.push(y);
                        transformed_keypoints.push(keypoint.clone().unwrap()[i + 2]);
                    }
                }

                // ID 추가
                keypoints.push(0.0);

                // Bbox 추가
                keypoints.push(bbox_x1);
                keypoints.push(bbox_y1);
                keypoints.push(bbox_x2);
                keypoints.push(bbox_y2);

                // keypoint 추가
                keypoints.extend(transformed_keypoints.iter().cloned());

                // 벡터를 문자열로 변환
                let data = keypoints
                    .iter()
                    .map(|&f| f.to_string())
                    .collect::<Vec<String>>()
                    .join(", ");
                let data = data.as_bytes();

                // 파일로 저장
                file.write_all(data)?;
                let result = file.write_all(b"\n");
                if let Err(e) = result {
                    println!("Error writing to file: {}", e);
                }
            }
        }
        print!("name:{:?} w:{:?},h:{:?} 완료 \n", file_name, width, height);
    }
    Ok(())
}
