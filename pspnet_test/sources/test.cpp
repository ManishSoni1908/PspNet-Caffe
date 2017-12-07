#include<caffe/caffe.hpp>
#include<caffe/common.hpp>
#include<caffe/util/upgrade_proto.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>


#define NO_CLASSES 41
#define THRESHOLD 0.9

using namespace std;
using namespace cv;
using namespace caffe;


string type2str(int type) {
    string r;

    uchar depth = type & CV_MAT_DEPTH_MASK;
    uchar chans = 1 + (type >> CV_CN_SHIFT);

    switch ( depth ) {
    case CV_8U:  r = "8U"; break;
    case CV_8S:  r = "8S"; break;
    case CV_16U: r = "16U"; break;
    case CV_16S: r = "16S"; break;
    case CV_32S: r = "32S"; break;
    case CV_32F: r = "32F"; break;
    case CV_64F: r = "64F"; break;
    default:     r = "User"; break;
    }

    r += "C";
    r += (chans+'0');

    return r;
}



int main(int argc, char** argv)
{
    const std::string object_names[]=
    {
        "Toilet_Brush"
        ,"Avery_Binder"
        ,"Balloons"
        ,"Band_Aid_Tape"
        ,"Bath_Sponge"
        ,"Black_Fashion_Gloves"
        ,"Burts_Bees_Baby_Wipes"
        ,"Colgate_Toothbrush_4PK"
        ,"Composition_Book"
        ,"Crayons"
        ,"Duct_Tape"
        ,"Epsom_Salts"
        ,"Expo_Eraser"
        ,"Fiskars_Scissors"
        ,"Flashlight"
        ,"Glue_Sticks"
        ,"Hand_Weight"
        ,"Hanes_Socks"
        ,"Hinged_Ruled_Index_Cards"
        ,"Ice_Cube_Tray"
        ,"Irish_Spring_Soap"
        ,"Laugh_Out_Loud_Jokes"
        ,"Marbles"
        ,"Measuring_Spoons"
        ,"Mesh_Cup"
        ,"Mouse_Traps"
        ,"Pie_Plates"
        ,"Plastic_Wine_Glass"
        ,"Poland_Spring_Water"
        ,"Reynolds_Wrap"
        ,"Robots_DVD"
        ,"Robots_Everywhere"
        ,"Scotch_Sponges"
        ,"Speed_Stick"
        ,"White_Facecloth"
        ,"Table_Cloth"
        ,"Tennis_Ball_Container"
        ,"Ticonderoga_Pencils"
        ,"Tissue_Box"
        ,"Windex"
    };





    cout << "Tesing code for pspnet" << endl;


    string test_proto = "/home/manish/tcs/arc_setup/utils/prototxt_testing/faddu_test.prototxt";
    string pre_trained_model = "/home/manish/tcs/arc_setup/trained_models/final__iter_23783.caffemodel";

    string image_path = "/home/manish/ValidationSet24/";


    ////////////setting caffe mode and device id ///////////////////////////////////

    Caffe::set_mode(Caffe::GPU);
    Caffe::SetDevice(0);


    ////defining net parameter and reading proto file ///////////////////////////////

    caffe::NetParameter net_param;
    caffe::ReadNetParamsFromTextFileOrDie(test_proto,&net_param);




    ///// intialize the whole net //////////////////
    caffe::Net<float> test_pspnet(net_param);


    ///// defining input and output blobs of network /////////////////////////////
    vector<Blob<float>* > input_blobs;
    vector<Blob<float>* > output_blobs;


    input_blobs = test_pspnet.input_blobs();
    output_blobs = test_pspnet.output_blobs();


    //////Reading the weights from pretrainde caffe model ///////////////////////////////

    test_pspnet.CopyTrainedLayersFrom(pre_trained_model);



    vector<int> input_blob_dim,output_blob_dim;


    cout << "input blob dimensions : " ;
    for(int i=0;i<input_blobs[0]->shape().size();i++)
    {
        input_blob_dim.push_back(input_blobs[0]->shape(i));
        cout << input_blobs[0]->shape(i) << "\t";
    }

    cout << endl;

    cout << "output blob dimensions : " ;
    for(int i=0;i<output_blobs[0]->shape().size();i++)
    {
        output_blob_dim.push_back(output_blobs[0]->shape(i));
        cout  << output_blobs[0]->shape(i) << "\t" ;
    }

    cout << endl;


    int it=3;
    while(it<12999)
    {
        cout << "Testing the segmentation" << endl;

        stringstream ss;

        ss << image_path << it << ".png";

        cv::Mat input_image;

        input_image = imread(ss.str());
        croppedImage(input_image);

        int input_image_width = input_image.cols;
        int input_image_height = input_image.rows;


        //// resize the input image according the network input
        cv::resize(input_image,input_image,cv::Size(input_blob_dim[2],input_blob_dim[3]));



        float* input_data_net = ((caffe::Blob<float>*)(input_blobs[0]))->mutable_cpu_data();

        cout << "input_image size" << input_image.rows << " " << input_image.cols << endl;

        float* input_data_net_ch1 = input_data_net;
        float* input_data_net_ch2 = input_data_net + input_image.rows*input_image.cols;
        float* input_data_net_ch3 = input_data_net + 2*input_image.rows*input_image.cols;




        /////////////// Feeding image into the network //////////////////////////////

        for(int i=0;i<input_image.rows ; i++)
            for(int j=0;j<input_image.cols ; j++)
            {
                //                float* pixel = input_image.at<cv::Vec3b>(i,j);

                (input_data_net_ch1 + i*input_image.cols)[j] = input_image.at<cv::Vec3b>(i,j)[0];
                (input_data_net_ch2 + i*input_image.cols)[j] = input_image.at<cv::Vec3b>(i,j)[1];
                (input_data_net_ch3 + i*input_image.cols)[j] = input_image.at<cv::Vec3b>(i,j)[2];

            }



        std::cout << "Forward passing\n";
        test_pspnet.ForwardFrom(0);
        std::cout << "Forward passing Done\n";




        float* output_data_net = ((caffe::Blob<float>* )(output_blobs[0]))->mutable_cpu_data();

        cv::Mat color_output_image;
        input_image.copyTo(color_output_image);



        cout << "image size" << color_output_image.size() <<endl;
        string ty =  type2str( color_output_image.type() );
        printf("Matrix: %s %dx%d \n", ty.c_str(), color_output_image.cols, color_output_image.rows );
        cv::cvtColor(color_output_image,color_output_image,CV_BGR2HSV);

        //        cout << "Type of image" << color_output_image. << endl;

        //        cout << "color output size " << color_output_image.rows << " " << color_output_image.cols << endl;

        cv::Mat mask  = cv::Mat::zeros(input_image.rows,input_image.cols,CV_8UC1);




        //////////// Analaysing the output /////////////////////////////////////////////////



        string object_name;

        for(int i=0 ; i< color_output_image.rows ; i++)
        {
            for(int j=0 ; j< color_output_image.cols ; j++)
            {
                float* prob_pixel = output_data_net + i* color_output_image.cols + j;
                int jump = color_output_image.cols * color_output_image.rows;



                float max = -1000000000.0;
                int index =0;

                for(int k=0;k<output_blob_dim[1];k++)
                {

                    float val = (prob_pixel + k*jump)[0];
                    if(val >  max)
                    {
                        max = val;
                        index = k;
                    }
                }


                if(index)
                {
                    object_name = object_names[index-1];

                    //                    cout << "index" << index << endl;
                    if(max > THRESHOLD)
                    {
                        //                                cout << (int)color_output_image.at<cv::Vec3b>(i,j)[0] <<" " ;
                        //                                color_output_image.at<cv::Vec3b>(i,j)[0] = /*(0)*(.67)*/10+.33*(input_image.at<cv::Vec3b>(i,j)[0]);
                        //                                color_output_image.at<cv::Vec3b>(i,j)[1] = /*(255*.77)*/70+.33*(input_image.at<cv::Vec3b>(i,j)[1]);
                        //                                color_output_image.at<cv::Vec3b>(i,j)[2] = /*(255*.77)*/ 10 +.33*(index+input_image.at<cv::Vec3b>(i,j)[2]);

                        color_output_image.at<cv::Vec3b>(i,j)[0] = 180*index/41;
                        color_output_image.at<cv::Vec3b>(i,j)[1] = 255;
                        color_output_image.at<cv::Vec3b>(i,j)[2] = 255;


                        mask.at<uchar>(i,j) = 255;
                    }

                }
            }
        }




        cv::cvtColor(color_output_image,color_output_image,CV_HSV2BGR);



        cv::resize(color_output_image,color_output_image,cv::Size(input_image_width,input_image_height));
        cv::resize(input_image,input_image,cv::Size(input_image_width,input_image_height));
        cv::resize(mask,mask,cv::Size(input_image_width,input_image_height));
        imshow("output_color_image",color_output_image);
        imshow("input_image",input_image);
        imshow("mask_genrated",mask);

        waitKey(0);

        it++;
    }

    return 0;
}
