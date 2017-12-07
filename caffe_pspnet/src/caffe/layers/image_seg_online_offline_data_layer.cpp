#include <fstream>  // NOLINT(readability/streams)
#include <iostream>  // NOLINT(readability/streams)
#include <string>
#include <utility>
#include <vector>
#include <algorithm>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/imgproc/imgproc.hpp>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/layers/image_seg_online_offline_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/io.hpp"
#include "caffe/util/math_functions.hpp"
#include "caffe/util/rng.hpp"

#include<signal.h>

namespace caffe {

template <typename Dtype>
ImageSegOnlineOfflineDataLayer<Dtype>::~ImageSegOnlineOfflineDataLayer<Dtype>() {
    this->StopInternalThread();

    flag_datagen_completed = true;
    for(int i=0;i<n_threads;i++)
        pthread_join(threads_clutter[i],NULL);

    delete threads_clutter;

    pthread_mutex_destroy(&mutex_rng);
    pthread_mutex_destroy(&mutex_thread_ids);
}


template <typename Dtype>
void* ImageSegOnlineOfflineDataLayer<Dtype>::generate_clutter(void* args) {

    int thread_id;

    pthread_mutex_lock(&mutex_thread_ids);
    {
        thread_id = thread_index++;
    }
    pthread_mutex_unlock(&mutex_thread_ids);

    int images_generated = 0;

    while( !flag_datagen_completed && (images_generated < n_images_per_thread))
    {
        int file_number;

        pthread_mutex_lock(&mutex_rng);
        {
            file_number = cv_rng.uniform(1,n_background_images+1);
        }
        pthread_mutex_unlock(&mutex_rng);


        std::stringstream img_file_name;
        img_file_name <<  background_images_dir.str() << file_number << ".png";

        int image_no = thread_id + n_threads * images_generated;

        //        cluttered_images[image_no].create(CROP_SIZE, CROP_SIZE,CV_8UC3);
        //        cluttered_masks[image_no].create(CROP_SIZE, CROP_SIZE,CV_8UC1);

        cv::Mat loc_image = cv::imread(img_file_name.str());

        cv::Rect roi;
        roi.x = (loc_image.cols -CROP_SIZE)/2.0;
        roi.y = (loc_image.rows -CROP_SIZE)/2.0;
        roi.width = CROP_SIZE;
        roi.height = CROP_SIZE;

        cv::Mat& cluttered_image = cluttered_images[image_no];
        cv::Mat& cluttered_mask = cluttered_masks[image_no];

        loc_image(roi).copyTo(cluttered_image);
        cluttered_mask = cv::Mat::zeros(CROP_SIZE, CROP_SIZE, CV_8UC1);

        // 3 levels of clutter low, medium, high
        //4 levels of occlusion 0,25,50,75

        int clutter_level;

        pthread_mutex_lock(&mutex_rng);
        {
            clutter_level = cv_rng(3);
        }
        pthread_mutex_unlock(&mutex_rng);

        int clutter_divisions[] = {3,4,5};


        int max_x_div = cluttered_image.cols / clutter_divisions[clutter_level];
        int max_y_div = cluttered_image.rows / clutter_divisions[clutter_level];

        unsigned char obj_processed[NUM_OBJECTS];

        memset(obj_processed,0,NUM_OBJECTS);


        for(int j=0;j<clutter_divisions[clutter_level]-1;j++)
            for(int k=0;k<clutter_divisions[clutter_level]-1;k++)
            {

                //                for(int l=0;l<2;l++)
                {
                    int object_id = cv_rng.uniform(1,NUM_OBJECTS+1)-1;

                    //                        //                            if(!obj_processed[object_id])
                    {
                        obj_processed[object_id] = 1;

                        int max_images = n_images[object_id];
                        int file_num;

                        pthread_mutex_lock(&mutex_rng);
                        {
                            file_num = cv_rng.uniform(1,max_images+1);
                        }
                        pthread_mutex_unlock(&mutex_rng);


                        int exact_file_num = file_num + file_offset[object_id];

                        std::stringstream clutter_sample_image_name;
                        clutter_sample_image_name << objects_cropped_objects_dir.str() << exact_file_num << ".png";


                        std::stringstream clutter_sample_mask_name;
                        clutter_sample_mask_name << objects_annotations_dir.str() <<  "mask_" << exact_file_num << ".png";

                        cv::Mat cluttered_sample_image = cv::imread(clutter_sample_image_name.str());
                        cv::Mat cluttered_sample_mask  = cv::imread(clutter_sample_mask_name.str());


                        cv::Mat cluttered_gray_mask;
                        cv::cvtColor(cluttered_sample_mask,cluttered_gray_mask,CV_BGR2GRAY);
                        cv::threshold(cluttered_gray_mask,cluttered_gray_mask,0,255,CV_THRESH_BINARY);


                        std::vector<std::vector<cv::Point2i> > clutter_contours;
                        cv::findContours(cluttered_gray_mask,clutter_contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_NONE);

                        cv::Rect clutter_bounding_rect;

                        if(clutter_contours.size())
                            clutter_bounding_rect =  cv::boundingRect(clutter_contours[0]);
                        else
                            continue;

                        cv::Point2i center;
                        center.x  = clutter_bounding_rect.x + clutter_bounding_rect.width/2.0f;
                        center.y  = clutter_bounding_rect.y + clutter_bounding_rect.height/2.0f;

                        cv::Point2i anchor;
                        anchor.x = max_x_div*(k+1);
                        anchor.y = max_y_div*(j+1);


                        clutter_bounding_rect =  cv::boundingRect(clutter_contours[0]);
                        for(int l=1;l<clutter_contours.size();l++)
                            clutter_bounding_rect |= cv::boundingRect(clutter_contours[l]);

                        //                                    //-----TO PREVENT FALSE OBJECT CROPPING AT THE BOUNDARIES
                        //                                    int pad = 2;
                        //                                    if(clutter_bounding_rect.x > 0)
                        //                                        clutter_bounding_rect.x -= pad/2;

                        //                                    if(clutter_bounding_rect.y > 0)
                        //                                        clutter_bounding_rect.y -= pad/2;

                        //                                    if(clutter_bounding_rect.x + clutter_bounding_rect.width < cluttered_sample_image.cols)
                        //                                        clutter_bounding_rect.width += pad;


                        //                                    if(clutter_bounding_rect.y + clutter_bounding_rect.height < cluttered_sample_image.rows)
                        //                                        clutter_bounding_rect.height += pad;


                        //                                    cv::Rect roi_to_copy;
                        //                                    roi_to_copy.x = anchor.x  - clutter_bounding_rect.width / 2;
                        //                                    roi_to_copy.y = anchor.y  - clutter_bounding_rect.height / 2;
                        //                                    roi_to_copy.width = clutter_bounding_rect.width;
                        //                                    roi_to_copy.height = clutter_bounding_rect.height;

                        //                                    if(roi_to_copy.x < 0)
                        //                                    {
                        //                                        //-----REDUCE THE WIDTH
                        //                                        roi_to_copy.width += roi_to_copy.x ;

                        //                                        clutter_bounding_rect.x -= roi_to_copy.x;
                        //                                        clutter_bounding_rect.width = roi_to_copy.width;

                        //                                        roi_to_copy.x = 0;

                        //                                    }

                        //                                    if(roi_to_copy.y < 0)
                        //                                    {

                        //                                        //-----REDUCE THE HEIGHT
                        //                                        roi_to_copy.height += roi_to_copy.y ;

                        //                                        clutter_bounding_rect.y -= roi_to_copy.y;
                        //                                        clutter_bounding_rect.height = roi_to_copy.height;

                        //                                        roi_to_copy.y = 0;
                        //                                    }

                        //                                    if(roi_to_copy.x + roi_to_copy.width > cluttered_sample_image.cols)
                        //                                    {

                        //                                        //-----REDUCE THE WIDTH
                        //                                        roi_to_copy.width -=  roi_to_copy.x + roi_to_copy.width - cluttered_sample_image.cols;
                        //                                        clutter_bounding_rect.width = roi_to_copy.width;
                        //                                    }

                        //                                    if(roi_to_copy.y + roi_to_copy.height > cluttered_sample_image.rows)
                        //                                    {
                        //                                        //-----REDUCE THE HEIGHT
                        //                                        roi_to_copy.height -=  roi_to_copy.y + roi_to_copy.height - cluttered_sample_image.rows;
                        //                                        clutter_bounding_rect.height = roi_to_copy.height;
                        //                                    }



                        //                                    cluttered_sample_image(clutter_bounding_rect).copyTo(cluttered_image(roi_to_copy),cluttered_sample_mask(clutter_bounding_rect));
                        //                                    cluttered_sample_mask(clutter_bounding_rect).copyTo(cluttered_mask(roi_to_copy),cluttered_sample_mask(clutter_bounding_rect));


                        for(int l=0;l<cluttered_sample_image.rows;l++)
                            for(int m=0;m<cluttered_sample_image.cols;m++)
                            {
                                unsigned char* cluttered_sample_mask_pixel = cluttered_sample_mask.data + l* cluttered_sample_mask.step[0] + m * cluttered_sample_mask.step[1];
                                unsigned char* cluttered_sample_image_pixel = cluttered_sample_image.data + l * cluttered_sample_image.step[0] + m * cluttered_sample_image.step[1];

                                if(cluttered_sample_mask_pixel[0])
                                {
                                    int loc_x = m - center.x + anchor.x;
                                    int loc_y = l - center.y + anchor.y;

                                    if(loc_x > -1 && loc_x < cluttered_sample_image.cols && loc_y > -1 && loc_y < cluttered_sample_image.rows )
                                    {
                                        unsigned char* cluttered_image_pixel = cluttered_image.data + loc_y * cluttered_image.step[0] + loc_x * cluttered_image.step[1];
                                        unsigned char* cluttered_mask_pixel = cluttered_mask.data + loc_y * cluttered_mask.step[0] + loc_x * cluttered_mask.step[1];


                                        cluttered_image_pixel[0] = cluttered_sample_image_pixel[0];
                                        cluttered_image_pixel[1] = cluttered_sample_image_pixel[1];
                                        cluttered_image_pixel[2] = cluttered_sample_image_pixel[2];

                                        cluttered_mask_pixel[0] = cluttered_sample_mask_pixel[0];
                                    }
                                }
                            }
                    }
                }
            }
        images_generated++;
        clutter_validity[image_no] = true;
    }
}


template <typename Dtype>
void ImageSegOnlineOfflineDataLayer<Dtype>::DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                                           const vector<Blob<Dtype>*>& top) {

    const int new_height = this->layer_param_.image_data_param().new_height();
    const int new_width  = this->layer_param_.image_data_param().new_width();
    const bool is_color  = this->layer_param_.image_data_param().is_color();
    const int label_type = this->layer_param_.image_data_param().label_type();
    string root_folder = this->layer_param_.image_data_param().root_folder();

    //-----READ PARAMS FROM FILES

    dataset_info << root_folder <<  "/dataset_info.yaml";
    background_images_dir << root_folder << "/../background_common/";
    objects_cropped_objects_dir << root_folder << "/objects/";
    objects_annotations_dir << root_folder << "/annotations/";

    n_background_images  = 0;

    boost::filesystem::directory_iterator dir_iterator(background_images_dir.str());
    while( dir_iterator != boost::filesystem::directory_iterator())
    {
        n_background_images++;
        *dir_iterator++;
    }

    cv::FileStorage file_dataset_info(dataset_info.str(),cv::FileStorage::READ);


    //----- ID MAPPING

    cv::Mat mat_id_mapping;
    file_dataset_info["ID_MAPPING"] >> mat_id_mapping;

    file_dataset_info["n_images_per_thread"] >> n_images_per_thread;
    file_dataset_info["n_threads"] >> n_threads;
    file_dataset_info["NUM_OBJECTS"] >> NUM_OBJECTS;
    file_dataset_info["CROP_SIZE"] >> CROP_SIZE;

    for(int i = 0; i < NUM_OBJECTS;i++)
    {
        int index = mat_id_mapping.at<unsigned char>(i,0);

        std::stringstream n_images_obj;
        n_images_obj << "n_images_" << index;

        int n_image;
        file_dataset_info[n_images_obj.str()] >> n_image;

        n_images.push_back(n_image);

        //-----COMPUTE OFFSET and ID MAPPING TO IMAGES FILES FOR EACH OBJECT

        int offset = 0;
        for(int j=0;j<i;j++)
            offset += n_images[j];

        file_offset.push_back(offset);

        int actual_id = mat_id_mapping.at<unsigned char>(i,0);
        int mapped_id = mat_id_mapping.at<unsigned char>(i,1);

        id_mapping[actual_id] = mapped_id;
    }

    max_clutter_images = n_threads * n_images_per_thread;

    cluttered_images = std::vector<cv::Mat>(max_clutter_images);
    cluttered_masks = std::vector<cv::Mat>(max_clutter_images);
    clutter_validity = std::vector<bool>(max_clutter_images);

    flag_datagen_completed = false;
    lines_clutter_id_ = 0;

    //-----

    //----- LAUNCH CLUTTER GENERATION THREADS

    thread_index = 0;

    pthread_mutex_init(&mutex_rng,NULL);
    pthread_mutex_init(&mutex_thread_ids,NULL);

    threads_clutter = new pthread_t[n_threads];

    for(int i=0;i<n_threads;i++)
        pthread_create(&threads_clutter[i],NULL,reinterpret_cast<void* (*)(void*)>(&ImageSegOnlineOfflineDataLayer::generate_clutter),this);

    //-----

    TransformationParameter transform_param = this->layer_param_.transform_param();
    CHECK(transform_param.has_mean_file() == false) <<
                                                       "ImageSegDataLayer does not support mean file";
    CHECK((new_height == 0 && new_width == 0) ||
          (new_height > 0 && new_width > 0)) << "Current implementation requires "
                                                "new_height and new_width to be set at the same time.";


    // Read the file with filenames and labels
    // const string& source = this->layer_param_.image_data_param().source();

    const string source = root_folder + "/list.txt";

    LOG(INFO) << "Opening file " << source;
    std::ifstream infile(source.c_str());

    string linestr;
    while (std::getline(infile, linestr)) {
        std::istringstream iss(linestr);
        string imgfn;
        iss >> imgfn;
        string segfn = "";
        if (label_type != ImageDataParameter_LabelType_NONE) {
            iss >> segfn;
        }
        lines_.push_back(std::make_pair(imgfn, segfn));
    }


    if (this->layer_param_.image_data_param().shuffle()) {
        // randomly shuffle data
        LOG(INFO) << "Shuffling data";
        const unsigned int prefetch_rng_seed = caffe_rng_rand();
        prefetch_rng_.reset(new Caffe::RNG(prefetch_rng_seed));
        ShuffleImages();
    }
    LOG(INFO) << "A total of " << lines_.size() << " images.";

    lines_id_ = 0;
    // Check if we would need to randomly skip a few data points
    if (this->layer_param_.image_data_param().rand_skip()) {
        unsigned int skip = caffe_rng_rand() %
                this->layer_param_.image_data_param().rand_skip();
        LOG(INFO) << "Skipping first " << skip << " data points.";
        CHECK_GT(lines_.size(), skip) << "Not enough points to skip";
        lines_id_ = skip;
    }
    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                      new_height, new_width, is_color);
    CHECK(cv_img.data) << "Could not load " << lines_[lines_id_].first;

    const int channels = cv_img.channels();
    const int height = cv_img.rows;
    const int width = cv_img.cols;
    // image
    //const int crop_size = this->layer_param_.transform_param().crop_size();
    int crop_width = 0;
    int crop_height = 0;
    CHECK((!transform_param.has_crop_size() && transform_param.has_crop_height() && transform_param.has_crop_width())
          || (!transform_param.has_crop_height() && !transform_param.has_crop_width()))
            << "Must either specify crop_size or both crop_height and crop_width.";
    if (transform_param.has_crop_size()) {
        crop_width = transform_param.crop_size();
        crop_height = transform_param.crop_size();
    }
    if (transform_param.has_crop_height() && transform_param.has_crop_width()) {
        crop_width = transform_param.crop_width();
        crop_height = transform_param.crop_height();
    }

    const int batch_size = this->layer_param_.image_data_param().batch_size();
    if (crop_width > 0 && crop_height > 0) {
        top[0]->Reshape(batch_size, channels, crop_height, crop_width);
        this->transformed_data_.Reshape(batch_size, channels, crop_height, crop_width);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].data_.Reshape(batch_size, channels, crop_height, crop_width);
        }

        //label
        top[1]->Reshape(batch_size, 1, crop_height, crop_width);
        this->transformed_label_.Reshape(batch_size, 1, crop_height, crop_width);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(batch_size, 1, crop_height, crop_width);
        }
    } else {
        top[0]->Reshape(batch_size, channels, height, width);
        this->transformed_data_.Reshape(batch_size, channels, height, width);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].data_.Reshape(batch_size, channels, height, width);
        }

        //label
        top[1]->Reshape(batch_size, 1, height, width);
        this->transformed_label_.Reshape(batch_size, 1, height, width);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(batch_size, 1, height, width);
        }
    }
    // image dimensions, for each image, stores (img_height, img_width)
    top[2]->Reshape(batch_size, 1, 1, 2);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].dim_.Reshape(batch_size, 1, 1, 2);
    }

    LOG(INFO) << "output data size: " << top[0]->num() << ","
              << top[0]->channels() << "," << top[0]->height() << ","
              << top[0]->width();
    // label
    LOG(INFO) << "output label size: " << top[1]->num() << ","
              << top[1]->channels() << "," << top[1]->height() << ","
              << top[1]->width();
    // image_dim
    LOG(INFO) << "output data_dim size: " << top[2]->num() << ","
              << top[2]->channels() << "," << top[2]->height() << ","
              << top[2]->width();
}

template <typename Dtype>
void ImageSegOnlineOfflineDataLayer<Dtype>::ShuffleImages() {
    caffe::rng_t* prefetch_rng =
            static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template <typename Dtype>
void ImageSegOnlineOfflineDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    Dtype* top_data     = batch->data_.mutable_cpu_data();
    Dtype* top_label    = batch->label_.mutable_cpu_data();
    Dtype* top_data_dim = batch->dim_.mutable_cpu_data();

    const int max_height = batch->data_.height();
    const int max_width  = batch->data_.width();

    ImageDataParameter image_data_param = this->layer_param_.image_data_param();
    const int batch_size = image_data_param.batch_size();
    const int new_height = image_data_param.new_height();
    const int new_width  = image_data_param.new_width();
    const int label_type = this->layer_param_.image_data_param().label_type();
    const int ignore_label = image_data_param.ignore_label();
    const bool is_color  = image_data_param.is_color();
    string root_folder   = image_data_param.root_folder();

    const int lines_size = lines_.size();
    int top_data_dim_offset;

    //  static int batch_no;



    for (int item_id = 0; item_id < batch_size; ++item_id) {
        top_data_dim_offset = batch->dim_.offset(item_id);

        std::vector<cv::Mat> cv_img_seg;

        // get a blob
        timer.Start();
        CHECK_GT(lines_size, lines_id_);

        int img_row, img_col;


        bool flag_read_from_clutter = rng_clutter_or_image.uniform(0,2);

        if(flag_read_from_clutter && lines_clutter_id_ < cluttered_images.size() && clutter_validity[lines_clutter_id_])
        {

            if(clutter_validity[lines_clutter_id_])
            {
                cv_img_seg.push_back(cv::Mat());
                cv_img_seg.push_back(cv::Mat());

                cluttered_images[lines_clutter_id_].copyTo(cv_img_seg[0]);
                cluttered_masks[lines_clutter_id_].copyTo(cv_img_seg[1]);

                //            std::stringstream str_image_file;
                //            str_image_file << root_folder << "/clutter_objects/" << lines_clutter_id_ << ".png";

                //            std::stringstream str_mask_file;
                //            str_mask_file << root_folder << "/clutter_annotations/mask_" << lines_clutter_id_ << ".png";

                //            cv_img_seg.push_back(PSP_ReadImageToCVMat(str_image_file.str(), new_height, new_width, is_color, &img_row, &img_col));


                //            cv_img_seg.push_back(ReadImageToCVMat(str_mask_file.str(), new_height, new_width, false));

                //            if (!cv_img_seg[0].data) {
                //                DLOG(INFO) << "Fail to load img: " << str_image_file.str();
                //            }

                //            if (!cv_img_seg[1].data) {
                //                DLOG(INFO) << "Fail to load seg: " << str_mask_file.str();
                //            }

                lines_clutter_id_++;
            }
        }
        else
        {
            cv_img_seg.push_back(PSP_ReadImageToCVMat(root_folder + lines_[lines_id_].first,
                                                      new_height, new_width, is_color, &img_row, &img_col));

            cv_img_seg.push_back(ReadImageToCVMat(root_folder + lines_[lines_id_].second,
                                                  new_height, new_width, false));


            lines_id_++;

            if (!cv_img_seg[0].data) {
                DLOG(INFO) << "Fail to load img: " << root_folder + lines_[lines_id_].first;
            }

            if (!cv_img_seg[1].data) {
                DLOG(INFO) << "Fail to load seg: " << root_folder + lines_[lines_id_].second;
            }
        }


        // TODO(jay): implement resize in ReadImageToCVMat
        // NOTE data_dim may not work when min_scale and max_scale != 1
        top_data_dim[top_data_dim_offset]     = static_cast<Dtype>(std::min(max_height, img_row));
        top_data_dim[top_data_dim_offset + 1] = static_cast<Dtype>(std::min(max_width, img_col));



        cv::Mat& mask = cv_img_seg[1];

        for(int i=0;i< mask.rows;i++)
            for(int j=0;j< mask.cols;j++)
            {
                unsigned char* mask_pixel = mask.data + i * mask.step[0] + j* mask.step[1];

                if(mask_pixel[0])
                    mask_pixel[0] = id_mapping[mask_pixel[0]];
            }

        read_time += timer.MicroSeconds();
        timer.Start();
        // Apply transformations (mirror, crop...) to the image
        int offset;
        offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset);

        offset = batch->label_.offset(item_id);
        this->transformed_label_.set_cpu_data(top_label + offset);

        this->data_transformer_->TransformImgAndSegOnline(cv_img_seg,
                                                          &(this->transformed_data_), &(this->transformed_label_),
                                                          ignore_label);


        cv::imshow("live",cv_img_seg[0]);
        cv::imshow("seg",5*cv_img_seg[1]);
        cv::waitKey(1);


        trans_time += timer.MicroSeconds();

        // go to the next std::vector<int>::iterator iter;
        if (lines_id_ >= lines_size) {
            // We have reached the end. Restart from the first.
            DLOG(INFO) << "Restarting data prefetching from start.";
            lines_id_ = 0;

            // flag_read_from_clutter_folder  =true;

            if (this->layer_param_.image_data_param().shuffle()) {
                ShuffleImages();
            }
        }

        if (lines_clutter_id_ >= max_clutter_images) {
            // We have reached the end. Restart from the first.
            DLOG(INFO) << "Restarting data prefetching from start.";
            lines_clutter_id_ = 0;
            // flag_read_from_clutter_folder  =true;
        }

    }
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(ImageSegOnlineOfflineDataLayer);
REGISTER_LAYER_CLASS(ImageSegOnlineOfflineData);

}  // namespace caffe
