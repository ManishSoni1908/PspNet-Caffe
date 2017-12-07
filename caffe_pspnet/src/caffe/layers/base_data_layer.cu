#include <vector>

#include "caffe/layers/base_data_layer.hpp"
#include<opencv2/opencv.hpp>

namespace caffe {

template <typename Dtype>
void BasePrefetchingDataLayer<Dtype>::Forward_gpu(
        const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {

    Batch<Dtype>* batch = prefetch_full_.pop("Data layer prefetch queue empty");
    //    std::cout << "POPED BATCH IS = " << (long int*)batch << "\n";

    // Reshape to loaded data.
    top[0]->ReshapeLike(batch->data_);
    // Copy the data
    caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
               top[0]->mutable_gpu_data());

    if (this->output_labels_) {
        // Reshape to loaded labels.
        top[1]->ReshapeLike(batch->label_);
        // Copy the labels.
        caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
                   top[1]->mutable_gpu_data());
    }

//    std::cout << "TOP BLOB DATA = ";
//    for(int i=3;i<7;i++)
//        std::cout << ((Blob<Dtype>*)top[1])->cpu_data()[i] << " ";
//    std::cout << "\n";

//    cv::Mat img = cv::Mat::zeros(512,512,CV_8UC3);

//    for(int i=0;i<512;i++)
//        for(int j=0;j<512;j++)
//        {
//            unsigned char* pixel = img.data + i* img.step[0] + j* img.step[1];
//            pixel[0] = top[0]->cpu_data()[i*512+j];
//            pixel[1] = top[0]->cpu_data()[img.cols*img.rows+i*512+j];
//            pixel[2] = top[0]->cpu_data()[img.cols*img.rows*2+i*512+j];
//        }

//    cv::Rect roi;
//    roi.x = 512*(top[1]->cpu_data()[3]);
//    roi.y = 512*(top[1]->cpu_data()[4]);
//    roi.width = 512*(-top[1]->cpu_data()[3] + top[1]->cpu_data()[5]);
//    roi.height = 512*(-top[1]->cpu_data()[4] + top[1]->cpu_data()[6]);

//    cv::rectangle(img,roi,cv::Scalar(255,0,0),2);
//    cv::imshow("top image",img);
//    cv::waitKey(1);


    // Ensure the copy is synchronous wrt the host, so that the next batch isn't
    // copied in meanwhile.
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
    prefetch_free_.push(batch);
}

template <typename Dtype>
void ImageDimPrefetchingDataLayer<Dtype>::Forward_gpu(
    const vector<Blob<Dtype>*>& bottom, const vector<Blob<Dtype>*>& top) {
  Batch<Dtype>* batch =
    BasePrefetchingDataLayer<Dtype>::prefetch_full_.pop("Data layer prefetch queue empty");
  // Reshape to loaded data.
  top[0]->ReshapeLike(batch->data_);
  // Copy the data
  caffe_copy(batch->data_.count(), batch->data_.gpu_data(),
      top[0]->mutable_gpu_data());
  if (this->output_labels_) {
    // Reshape to loaded labels.
    top[1]->ReshapeLike(batch->label_);
    // Copy the labels.
    caffe_copy(batch->label_.count(), batch->label_.gpu_data(),
        top[1]->mutable_gpu_data());
  }
  if (output_data_dim_) {
    // Reshape to loaded labels.
    top[2]->ReshapeLike(batch->dim_);
    // Copy the labels.
    caffe_copy(batch->dim_.count(), batch->dim_.gpu_data(),
        top[2]->mutable_gpu_data());
  }
  // Ensure the copy is synchronous wrt the host, so that the next batch isn't
  // copied in meanwhile.
  CUDA_CHECK(cudaStreamSynchronize(cudaStreamDefault));
  BasePrefetchingDataLayer<Dtype>::prefetch_free_.push(batch);
}


INSTANTIATE_LAYER_GPU_FORWARD(BasePrefetchingDataLayer);
INSTANTIATE_LAYER_GPU_FORWARD(ImageDimPrefetchingDataLayer);

}  // namespace caffe
