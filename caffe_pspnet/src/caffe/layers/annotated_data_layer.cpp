#ifdef USE_OPENCV
#include <opencv2/core/core.hpp>
#endif  // USE_OPENCV
#include <stdint.h>

#include <algorithm>
#include <map>
#include <vector>

#include "caffe/data_transformer.hpp"
#include "caffe/layers/annotated_data_layer.hpp"
#include "caffe/util/benchmark.hpp"
#include "caffe/util/sampler.hpp"
#include<opencv2/opencv.hpp>
#include "caffe/util/rng.hpp"

using namespace std;

namespace caffe {

void image_seg_to_annotated_datum(std::string& root_folder,
                                  vector<std::pair<std::string, std::string> >& lines_,
                                  int lines_id_,
                                  AnnotatedDatum& annotated_datum,
                                  int n_classes)
{
    // Read an image, and use it to initialize the top blob.
    cv::Mat cv_img = ReadImageToCVMat(root_folder + lines_[lines_id_].first,true);
    CHECK(cv_img.data) << "Fail to load img: " << root_folder + lines_[lines_id_].first;

    cv::Mat cv_seg = ReadImageToCVMat(root_folder + lines_[lines_id_].second,false);
    CHECK(cv_img.data) << "Fail to load seg: " << root_folder + lines_[lines_id_].second;


    cv::Rect roi;

    roi.x = (cv_img.cols-512)/2;
    roi.y = (cv_img.rows-512)/2;
    roi.width = 512;
    roi.height = 512;

    cv_img = cv_img(roi);
    cv_seg = cv_seg(roi);

    cv::Mat rect_img;
    cv_img.copyTo(rect_img);

    annotated_datum.clear_annotation_group();
    annotated_datum.clear_datum();


//    for(int i=1;i<n_classes;i++)
//    {
//        cv::Mat object_mask = (cv_seg == i);

//        std::vector<std::vector<cv::Point2i> > contours;
//        cv::findContours(object_mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

//        cv::Rect bounding_rect;

//        if(contours.size())
//        {
//            std::vector<cv::Point2i> points;

//            for(int contour = 0;contour < contours.size();contour++)
//            {
//                for(int n_points=0;n_points<contours[contour].size();n_points++)
//                    points.push_back(contours[contour][n_points]);
//            }

//            bounding_rect =  cv::boundingRect(points);
//            AnnotationGroup* anno_group =  annotated_datum.add_annotation_group();
//            anno_group->set_group_label(i);

//            Annotation* anno = anno_group->add_annotation();
//            anno->set_instance_id(1);
//            anno->mutable_bbox()->set_xmin(((float)bounding_rect.x) / cv_img.cols);
//            anno->mutable_bbox()->set_ymin(((float)bounding_rect.y) / cv_img.rows);
//            anno->mutable_bbox()->set_xmax(((float)bounding_rect.x + bounding_rect.width + 1.0f) / cv_img.cols);
//            anno->mutable_bbox()->set_ymax(((float)bounding_rect.y + bounding_rect.height + 1.0f) / cv_img.rows);
//            anno->mutable_bbox()->set_label(i);
//            anno->mutable_bbox()->set_difficult(true);

//            bounding_rect.x = 512*anno->bbox().xmin();
//            bounding_rect.y = 512*anno->bbox().ymin();
//            bounding_rect.width = 512*(anno->bbox().xmax() - anno->bbox().xmin());
//            bounding_rect.height = 512*(anno->bbox().ymax() - anno->bbox().ymin());

////            std::cout << "ANNO BOX = " << anno->bbox().xmin() << "  " << anno->bbox().ymin() <<
////                         "  " << anno->bbox().xmax() << "  " << anno->bbox().ymax() << "\n";
//            cv::rectangle(rect_img,bounding_rect,cv::Scalar(0,255,0),2);


//        }

//    }


    AnnotationGroup* anno_group =  annotated_datum.add_annotation_group();
    anno_group->set_group_label(1);

    int object_label = 1;

    int instance_id = 0;
    for(int i=1;i<n_classes;i++)
    {
        cv::Mat object_mask = (cv_seg == i);

        std::vector<std::vector<cv::Point2i> > contours;
        cv::findContours(object_mask,contours,cv::RETR_EXTERNAL,cv::CHAIN_APPROX_SIMPLE);

        cv::Rect bounding_rect;

        if(contours.size())
        {
            std::vector<cv::Point2i> points;

            for(int contour = 0;contour < contours.size();contour++)
            {
                for(int n_points=0;n_points<contours[contour].size();n_points++)
                    points.push_back(contours[contour][n_points]);
            }

            bounding_rect =  cv::boundingRect(points);

            Annotation* anno = anno_group->add_annotation();
            anno->set_instance_id(instance_id++);
            anno->mutable_bbox()->set_xmin(((float)bounding_rect.x) / cv_img.cols);
            anno->mutable_bbox()->set_ymin(((float)bounding_rect.y) / cv_img.rows);
            anno->mutable_bbox()->set_xmax(((float)bounding_rect.x + bounding_rect.width + 1.0f) / cv_img.cols);
            anno->mutable_bbox()->set_ymax(((float)bounding_rect.y + bounding_rect.height + 1.0f) / cv_img.rows);
            anno->mutable_bbox()->set_label(object_label);
            anno->mutable_bbox()->set_difficult(true);

            bounding_rect.x = 512*anno->bbox().xmin();
            bounding_rect.y = 512*anno->bbox().ymin();
            bounding_rect.width = 512*(anno->bbox().xmax() - anno->bbox().xmin());
            bounding_rect.height = 512*(anno->bbox().ymax() - anno->bbox().ymin());

//            std::cout << "ANNO BOX = " << anno->bbox().xmin() << "  " << anno->bbox().ymin() <<
//                         "  " << anno->bbox().xmax() << "  " << anno->bbox().ymax() << "\n";
            cv::rectangle(rect_img,bounding_rect,cv::Scalar(0,255,0),2);


        }

    }

    cv::imshow("cv_img",rect_img);
    cv::waitKey(1);
    annotated_datum.set_type(AnnotatedDatum_AnnotationType_BBOX);
    EncodeCVMatToDatum(cv_img,".jpg", annotated_datum.mutable_datum());

}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::AnnotatedDataLayer(const LayerParameter& param)
    : BasePrefetchingDataLayer<Dtype>(param)
{
}

template <typename Dtype>
AnnotatedDataLayer<Dtype>::~AnnotatedDataLayer() {
    this->StopInternalThread();
}

template <typename Dtype>
void AnnotatedDataLayer<Dtype>::DataLayerSetUp(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
    //    const int batch_size = this->layer_param_.data_param().batch_size();

    const int batch_size = this->layer_param_.image_data_param().batch_size();
    const AnnotatedDataParameter& anno_data_param =
            this->layer_param_.annotated_data_param();
    for (int i = 0; i < anno_data_param.batch_sampler_size(); ++i) {
        batch_samplers_.push_back(anno_data_param.batch_sampler(i));
    }
    label_map_file_ = anno_data_param.label_map_file();
    // Make sure dimension is consistent within batch.
    const TransformationParameter& transform_param =
            this->layer_param_.transform_param();
    if (transform_param.has_resize_param()) {
        if (transform_param.resize_param().resize_mode() ==
                ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
            CHECK_EQ(batch_size, 1)
                    << "Only support batch size of 1 for FIT_SMALL_SIZE.";
        }
    }

    ///--------------- INITIALIZING IMAGE AND SEG DATA READER

    string root_folder = this->layer_param_.image_data_param().root_folder();
    const int label_type = this->layer_param_.image_data_param().label_type();
    const int n_classes = this->layer_param_.image_data_param().n_classes();

    // Read the file with filenames and labels
    const string& source = this->layer_param_.image_data_param().source();
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

    ///--------------------ENDS
    // Read a data point, and use it to initialize the top blob.
    AnnotatedDatum anno_datum;

    image_seg_to_annotated_datum(root_folder,lines_,lines_id_,anno_datum,n_classes);

    // Use data_transformer to infer the expected blob shape from anno_datum.
    vector<int> top_shape =
            this->data_transformer_->InferBlobShape(anno_datum.datum());
    this->transformed_data_.Reshape(top_shape);
    // Reshape top[0] and prefetch_data according to the batch_size.
    top_shape[0] = batch_size;
    top[0]->Reshape(top_shape);
    for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
        this->prefetch_[i].data_.Reshape(top_shape);
    }
    LOG(INFO) << "output data size: " << top[0]->num() << ","
              << top[0]->channels() << "," << top[0]->height() << ","
              << top[0]->width();
    // label
    if (this->output_labels_) {
        has_anno_type_ = anno_datum.has_type() || anno_data_param.has_anno_type();
        vector<int> label_shape(4, 1);
        if (has_anno_type_) {
            anno_type_ = anno_datum.type();
            if (anno_data_param.has_anno_type()) {
                // If anno_type is provided in AnnotatedDataParameter, replace
                // the type stored in each individual AnnotatedDatum.
                LOG(WARNING) << "type stored in AnnotatedDatum is shadowed.";
                anno_type_ = anno_data_param.anno_type();
            }
            // Infer the label shape from anno_datum.AnnotationGroup().
            int num_bboxes = 0;
            if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
                // Since the number of bboxes can be different for each image,
                // we store the bbox information in a specific format. In specific:
                // All bboxes are stored in one spatial plane (num and channels are 1)
                // And each row contains one and only one box in the following format:
                // [item_id, group_label, instance_id, xmin, ymin, xmax, ymax, diff]
                // Note: Refer to caffe.proto for details about group_label and
                // instance_id.
                for (int g = 0; g < anno_datum.annotation_group_size(); ++g) {
                    num_bboxes += anno_datum.annotation_group(g).annotation_size();
                }
                label_shape[0] = 1;
                label_shape[1] = 1;
                // BasePrefetchingDataLayer<Dtype>::LayerSetUp() requires to call
                // cpu_data and gpu_data for consistent prefetch thread. Thus we make
                // sure there is at least one bbox.
                label_shape[2] = std::max(num_bboxes, 1);
                label_shape[3] = 8;
            } else {
                LOG(FATAL) << "Unknown annotation type.";
            }
        } else {
            label_shape[0] = batch_size;
        }
        top[1]->Reshape(label_shape);
        for (int i = 0; i < this->PREFETCH_COUNT; ++i) {
            this->prefetch_[i].label_.Reshape(label_shape);
        }
    }
}



template <typename Dtype>
void AnnotatedDataLayer<Dtype>::ShuffleImages() {
    caffe::rng_t* prefetch_rng =
            static_cast<caffe::rng_t*>(prefetch_rng_->generator());
    shuffle(lines_.begin(), lines_.end(), prefetch_rng);
}

// This function is called on prefetch thread
template<typename Dtype>
void AnnotatedDataLayer<Dtype>::load_batch(Batch<Dtype>* batch) {
    //    std::cout << "LOADING BATCH\n";

    CPUTimer batch_timer;
    batch_timer.Start();
    double read_time = 0;
    double trans_time = 0;
    CPUTimer timer;
    CHECK(batch->data_.count());
    CHECK(this->transformed_data_.count());

    // Reshape according to the first anno_datum of each batch
    // on single input batches allows for inputs of varying dimension.
    //    const int batch_size = this->layer_param_.data_param().batch_size();
    const int batch_size = this->layer_param_.image_data_param().batch_size();

    const AnnotatedDataParameter& anno_data_param =
            this->layer_param_.annotated_data_param();
    const TransformationParameter& transform_param =
            this->layer_param_.transform_param();

    // IMAGE SEG STYLE
    string root_folder = this->layer_param_.image_data_param().root_folder();
    const int lines_size = lines_.size();
    const int n_classes = this->layer_param_.image_data_param().n_classes();

    AnnotatedDatum anno_datum;
    image_seg_to_annotated_datum(root_folder,lines_,lines_id_,anno_datum,n_classes);


    // Use data_transformer to infer the expected blob shape from anno_datum.
    vector<int> top_shape =
            this->data_transformer_->InferBlobShape(anno_datum.datum());
    this->transformed_data_.Reshape(top_shape);
    // Reshape batch according to the batch_size.
    top_shape[0] = batch_size;
    batch->data_.Reshape(top_shape);

    Dtype* top_data = batch->data_.mutable_cpu_data();
    Dtype* top_label = NULL;  // suppress warnings about uninitialized variables
    if (this->output_labels_ && !has_anno_type_) {
        top_label = batch->label_.mutable_cpu_data();
    }


    // Store transformed annotation.
    map<int, vector<AnnotationGroup> > all_anno;
    int num_bboxes = 0;

    for (int item_id = 0; item_id < batch_size; ++item_id) {
        timer.Start();
        // get a anno_datum

        AnnotatedDatum anno_datum;

        // IMAGE SEG STYLE
        image_seg_to_annotated_datum(root_folder,lines_,lines_id_,anno_datum,n_classes);
        //        AnnotatedDatum& anno_datum = *(reader_.full().pop("Waiting for data"));

        read_time += timer.MicroSeconds();
        timer.Start();
        AnnotatedDatum distort_datum;
        AnnotatedDatum* expand_datum = NULL;
        if (transform_param.has_distort_param()) {
            distort_datum.CopyFrom(anno_datum);
            this->data_transformer_->DistortImage(anno_datum.datum(),
                                                  distort_datum.mutable_datum());
            if (transform_param.has_expand_param()) {
                expand_datum = new AnnotatedDatum();
                this->data_transformer_->ExpandImage(distort_datum, expand_datum);
            } else {
                expand_datum = &distort_datum;
            }
        } else {
            if (transform_param.has_expand_param()) {
                expand_datum = new AnnotatedDatum();
                this->data_transformer_->ExpandImage(anno_datum, expand_datum);
            } else {
                expand_datum = &anno_datum;
            }
        }
        AnnotatedDatum* sampled_datum = NULL;
        bool has_sampled = false;
        if (batch_samplers_.size() > 0) {
            // Generate sampled bboxes from expand_datum.
            vector<NormalizedBBox> sampled_bboxes;
            GenerateBatchSamples(*expand_datum, batch_samplers_, &sampled_bboxes);
            if (sampled_bboxes.size() > 0) {
                // Randomly pick a sampled bbox and crop the expand_datum.
                int rand_idx = caffe_rng_rand() % sampled_bboxes.size();
                sampled_datum = new AnnotatedDatum();
                this->data_transformer_->CropImage(*expand_datum,
                                                   sampled_bboxes[rand_idx],
                                                   sampled_datum);
                has_sampled = true;
            } else {
                sampled_datum = expand_datum;
            }
        } else {
            sampled_datum = expand_datum;
        }
        CHECK(sampled_datum != NULL);
        timer.Start();
        vector<int> shape =
                this->data_transformer_->InferBlobShape(sampled_datum->datum());
        if (transform_param.has_resize_param()) {
            if (transform_param.resize_param().resize_mode() ==
                    ResizeParameter_Resize_mode_FIT_SMALL_SIZE) {
                this->transformed_data_.Reshape(shape);
                batch->data_.Reshape(shape);
                top_data = batch->data_.mutable_cpu_data();
            } else {
                CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                                 shape.begin() + 1));
            }
        } else {
            CHECK(std::equal(top_shape.begin() + 1, top_shape.begin() + 4,
                             shape.begin() + 1));
        }
        // Apply data transformations (mirror, scale, crop...)
        int offset = batch->data_.offset(item_id);
        this->transformed_data_.set_cpu_data(top_data + offset);
        vector<AnnotationGroup> transformed_anno_vec;
        if (this->output_labels_) {
            if (has_anno_type_) {
                // Make sure all data have same annotation type.
                CHECK(sampled_datum->has_type()) << "Some datum misses AnnotationType.";
                if (anno_data_param.has_anno_type()) {
                    sampled_datum->set_type(anno_type_);
                } else {
                    CHECK_EQ(anno_type_, sampled_datum->type()) <<
                                                                   "Different AnnotationType.";
                }
                // Transform datum and annotation_group at the same time
                transformed_anno_vec.clear();
                this->data_transformer_->Transform(*sampled_datum,
                                                   &(this->transformed_data_),
                                                   &transformed_anno_vec);
                if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
                    // Count the number of bboxes.
                    for (int g = 0; g < transformed_anno_vec.size(); ++g) {
                        num_bboxes += transformed_anno_vec[g].annotation_size();
                    }
                } else {
                    LOG(FATAL) << "Unknown annotation type.";
                }
                all_anno[item_id] = transformed_anno_vec;
            } else {
                this->data_transformer_->Transform(sampled_datum->datum(),
                                                   &(this->transformed_data_));
                // Otherwise, store the label from datum.
                CHECK(sampled_datum->datum().has_label()) << "Cannot find any label.";
                top_label[item_id] = sampled_datum->datum().label();
            }
        } else {
            this->data_transformer_->Transform(sampled_datum->datum(),
                                               &(this->transformed_data_));
        }
        // clear memory
        if (has_sampled) {
            delete sampled_datum;
        }
        if (transform_param.has_expand_param()) {
            delete expand_datum;
        }
        trans_time += timer.MicroSeconds();

        // go to the next std::vector<int>::iterator iter;
        lines_id_++;
        if (lines_id_ >= lines_size) {
            // We have reached the end. Restart from the first.
            DLOG(INFO) << "Restarting data prefetching from start.";
            lines_id_ = 0;
            if (this->layer_param_.image_data_param().shuffle()) {
                ShuffleImages();
            }
        }
        //        reader_.free().push(const_cast<AnnotatedDatum*>(&anno_datum));
    }

    // Store "rich" annotation if needed.
    if (this->output_labels_ && has_anno_type_) {
        vector<int> label_shape(4);
        if (anno_type_ == AnnotatedDatum_AnnotationType_BBOX) {
            label_shape[0] = 1;
            label_shape[1] = 1;
            label_shape[3] = 8;
            if (num_bboxes == 0) {
                // Store all -1 in the label.
                label_shape[2] = 1;
                batch->label_.Reshape(label_shape);
                caffe_set<Dtype>(8, -1, batch->label_.mutable_cpu_data());
            } else {
                // Reshape the label and store the annotation.
                label_shape[2] = num_bboxes;
                batch->label_.Reshape(label_shape);
                top_label = batch->label_.mutable_cpu_data();
                int idx = 0;
                for (int item_id = 0; item_id < batch_size; ++item_id) {
                    const vector<AnnotationGroup>& anno_vec = all_anno[item_id];
                    for (int g = 0; g < anno_vec.size(); ++g) {
                        const AnnotationGroup& anno_group = anno_vec[g];
                        for (int a = 0; a < anno_group.annotation_size(); ++a) {
                            const Annotation& anno = anno_group.annotation(a);
                            const NormalizedBBox& bbox = anno.bbox();
                            top_label[idx++] = item_id;
                            top_label[idx++] = anno_group.group_label();
                            top_label[idx++] = anno.instance_id();
                            top_label[idx++] = bbox.xmin();
                            top_label[idx++] = bbox.ymin();
                            top_label[idx++] = bbox.xmax();
                            top_label[idx++] = bbox.ymax();
                            top_label[idx++] = bbox.difficult();
                        }
                    }
                }

                //                std::cout << "FILLING RECTANGLES\n";
            }
        } else {
            LOG(FATAL) << "Unknown annotation type.";
        }
    }

    //    std::cout << "BATCH BLOB DATA = ";
    //    for(int i=3;i<7;i++)
    //    std::cout << batch->label_.cpu_data()[i] << " ";
    //    std::cout << "\n";

    timer.Stop();
    batch_timer.Stop();
    DLOG(INFO) << "Prefetch batch: " << batch_timer.MilliSeconds() << " ms.";
    DLOG(INFO) << "     Read time: " << read_time / 1000 << " ms.";
    DLOG(INFO) << "Transform time: " << trans_time / 1000 << " ms.";
}

INSTANTIATE_CLASS(AnnotatedDataLayer);
REGISTER_LAYER_CLASS(AnnotatedData);

}  // namespace caffe
