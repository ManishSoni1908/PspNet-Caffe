#ifndef CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
#define CAFFE_IMAGE_SEG_DATA_LAYER_HPP_

#include <string>
#include <utility>
#include <vector>

#include "caffe/blob.hpp"
#include "caffe/data_transformer.hpp"
#include "caffe/internal_thread.hpp"
#include "caffe/layer.hpp"
#include "caffe/layers/base_data_layer.hpp"
#include "caffe/proto/caffe.pb.h"
#include<opencv2/opencv.hpp>

namespace caffe {


template <typename Dtype>
class ImageSegOnlineOfflineDataLayer : public ImageDimPrefetchingDataLayer<Dtype> {
public:
    explicit ImageSegOnlineOfflineDataLayer(const LayerParameter& param)
        : ImageDimPrefetchingDataLayer<Dtype>(param) {}
    virtual ~ImageSegOnlineOfflineDataLayer();
    virtual void DataLayerSetUp(const vector<Blob<Dtype>*>& bottom,
                                const vector<Blob<Dtype>*>& top);

    virtual inline const char* type() const { return "ImageSegData"; }
    virtual inline int ExactNumBottomBlobs() const { return 0; }
    virtual inline int ExactNumTopBlobs() const { return 3; }
    virtual inline bool AutoTopBlobs() const { return true; }

protected:
    virtual void ShuffleImages();
    virtual void load_batch(Batch<Dtype>* batch);


    bool flag_datagen_completed;
    int n_background_images;
    int max_clutter_images;
    int n_threads;
    int NUM_OBJECTS;
    int CROP_SIZE;
    int n_images_per_thread;
    int lines_clutter_id_;

    std::vector<int> n_images;
    std::vector<int> file_offset;

    std::vector<cv::Mat> cluttered_images;
    std::vector<cv::Mat> cluttered_masks;
    std::vector<bool> clutter_validity;

    std::stringstream dataset_info;
    std::stringstream background_images_dir;
    std::stringstream objects_cropped_objects_dir;
    std::stringstream objects_annotations_dir;

    std::map<int, int> id_mapping;

    cv::RNG cv_rng;

    void* generate_clutter(void* args);
    int thread_index;
    pthread_t* threads_clutter;
    pthread_mutex_t mutex_rng;
    pthread_mutex_t mutex_thread_ids;

    cv::RNG rng_clutter_or_image;

    Blob<Dtype> transformed_label_;
    shared_ptr<Caffe::RNG> prefetch_rng_;
    vector<std::pair<std::string, std::string> > lines_;
    int lines_id_;
};

}  // namespace caffe

#endif  // CAFFE_IMAGE_SEG_DATA_LAYER_HPP_
