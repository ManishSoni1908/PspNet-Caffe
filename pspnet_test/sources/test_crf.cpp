#include<caffe/caffe.hpp>
#include<caffe/common.hpp>
#include<caffe/util/upgrade_proto.hpp>
#include<opencv2/opencv.hpp>
#include<iostream>


#define NO_CLASSES 41

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




void croppedImage(cv::Mat &Image)
{
    cout << "Enter bin id " << endl;
    int bin_id =2;
    //    cin >> bin_id;

    stringstream ss;
    ss << "/home/manish/Music/mask" /*<< bin_id */<< ".png";


    cv::Mat mask = cv::imread(ss.str());



    //    imshow("mask",mask);
    //    waitKey(0);

    cv::bitwise_and(Image,mask,Image);

    cvtColor(mask,mask,CV_BGR2GRAY);
    vector<vector<Point> > v;

    // Finds contours
    findContours(mask,v,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

    // Finds the contour with the largest area
    int area = 0;
    int idx;
    for(int i=0; i<v.size();i++) {
        if(area < v[i].size())
            idx = i;
    }


    Rect rect = boundingRect(v[idx]);
    Point pt1, pt2;
    pt1.x = rect.x;
    pt1.y = rect.y;
    pt2.x = rect.x + rect.width;
    pt2.y = rect.y + rect.height;
    // Draws the rect in the original image and show it
    Image = Image(rect);


    //    Image.copyTo(Image,mask);
    //    cv::imshow("cropped image",Image);
    //    waitKey(0);
    ss.str("");

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

    //    string test_proto = "/home/manish/tcs/arc_setup/utils/prototxt_testing/faddu_test.prototxt";
    //    string pre_trained_model = "/home/manish/tcs/arc_setup/trained_models/new_test__iter_1161.caffemodel";

    string test_proto = "/home/manish/tcs/arc_setup/utils/prototxt_testing/faddu_test_crf.prototxt";
    string pre_trained_model = "/home/manish/final__iter_50000.caffemodel";

//        string test_proto = "/home/manish/tcs/arc_setup/utils/prototxt_testing/pspnet50_ADE20K_473_crf.prototxt";
    //    string pre_trained_model = "/home/manish/41_object__iter_25000.caffemodel";

    //    string image_path = "/home/manish/arc17'vision/apc_new_objects_tote_data_foscam/cropped_objects/";

    string image_path = "/home/manish/ValidationSet24/";
//        string image_path = "/home/manish/Pictures/cropped_images/ValidationSet18/";


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



    vector<int> input_blob_dim,output_blob_dim,crf_input_blob_dim;


    cout << "input blob dimensions : " ;
    for(int i=0;i<input_blobs[0]->shape().size();i++)
    {
        input_blob_dim.push_back(input_blobs[0]->shape(i));
        cout << input_blobs[0]->shape(i) << "\t";
    }


    cout << "input blob for crf dimensions : " ;
    for(int i=0;i<input_blobs[0]->shape().size();i++)
    {
        crf_input_blob_dim.push_back(input_blobs[1]->shape(i));
        cout << input_blobs[1]->shape(i) << "\t";
    }

    cout << endl;

    cout << "output blob dimensions : " ;
    for(int i=0;i<output_blobs[0]->shape().size();i++)
    {
        output_blob_dim.push_back(output_blobs[0]->shape(i));
        cout  << output_blobs[0]->shape(i) << "\t" ;
    }

    cout << endl;

    int ids_avilable[14] = {25,18,20,40,12,6,3,15,29,4,9,2,23,10};



    int it=3;
    while(it<12999)
    {
        cout << "Testing the segmentation" << endl;

        stringstream ss;
        //                ss << image_path << it << ".jpg";
        ss << image_path << it << ".png";

        cv::Mat input_image;

        input_image = imread(ss.str());
        croppedImage(input_image);

        int input_image_width = input_image.cols;
        int input_image_height = input_image.rows;


        //// resize the input image according the network input
        cv::resize(input_image,input_image,cv::Size(input_blob_dim[2],input_blob_dim[3]));



        float* input_data_net = ((caffe::Blob<float>*)(input_blobs[0]))->mutable_cpu_data();

        //        float* output_data_net = ((caffe::Blob<float>* )(output_blobs[0]))->mutable_cpu_data();

        //        imshow("input_image",input_image);
        //        waitKey(0);


        cout << "input_image size" << input_image.rows << " " << input_image.cols << endl;

        float* input_data_net_ch1 = input_data_net;
        float* input_data_net_ch2 = input_data_net + input_image.rows*input_image.cols;
        float* input_data_net_ch3 = input_data_net + 2*input_image.rows*input_image.cols;



        float* input_crf = ((caffe::Blob<float>*)(input_blobs[1]))->mutable_cpu_data();

        input_crf[0] = input_blob_dim[2];
        input_crf[1] = input_blob_dim[3];




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


//                for(int t=0;t<14;t++)
//                {
                    if(index /*== ids_avilable[t]*/)
                    {
                        object_name = object_names[index-1];

                        //                    cout << "index" << index << endl;
                        if(max > 0.89)
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

//                    }
                }
            }
        }

        //               cv::addWeighted()

        //        ///////////// coloring of background ////////////////////////

        //        string object_name;


        //        for(int i=0 ; i< color_output_image.rows ; i++)
        //        {
        //            for(int j=0 ; j< color_output_image.cols ; j++)
        //            {
        //                float* prob_pixel = output_data_net + i* color_output_image.cols + j;
        //                int jump = color_output_image.cols * color_output_image.rows;

        //                float max = -1000000000.0;
        //                int index =0;

        //                for(int k=0;k<output_blob_dim[1];k++)
        //                {

        //                    float val = (prob_pixel + k*jump)[0];
        //                    if(val >  max)
        //                    {
        //                        max = val;
        //                        index = k;
        //                    }
        //                }


        //                if(index==0)
        //                {
        //                    object_name = "background";

        //                    //                    cout << "index" << index << endl;
        //                    if(max > 0.50)
        //                    {
        //                        color_output_image.at<cv::Vec3b>(i,j)[0] = 180*index/NO_CLASSES;
        //                        color_output_image.at<cv::Vec3b>(i,j)[1] = 1;
        //                        color_output_image.at<cv::Vec3b>(i,j)[2] = 2;

        //                        mask.at<uchar>(i,j) = 255;
        //                    }

        //                }
        //            }
        //        }


        cv::cvtColor(color_output_image,color_output_image,CV_HSV2BGR);



        //                cout << "testing done" << endl;


        vector<vector<Point> > v;

        //        // Finds contours
        //        findContours(mask,v,CV_RETR_LIST,CV_CHAIN_APPROX_NONE);

        //        // Finds the contour with the largest area
        //        int area = 0;
        //        int idx;
        //        for(int i=0; i<v.size();i++) {
        //            if(area < v[i].size())
        //                idx = i;
        //        }

        // Calculates the bounding rect of the largest area contour
        //        Rect rect = boundingRect(v[idx]);
        //        Point pt1, pt2;
        //        pt1.x = rect.x;
        //        pt1.y = rect.y;
        //        pt2.x = rect.x + rect.width;
        //        pt2.y = rect.y + rect.height;
        //        // Draws the rect in the original image and show it
        //        rectangle(color_output_image, pt1, pt2, CV_RGB(255,255,0), 2);


        //        int fontFace =FONT_HERSHEY_SIMPLEX;
        //        //        cv::putText(color_output_image,object_name,pt1,fontFace,1,cv::Scalar(255,255,255));



        //        stringstream ss1;
        //        ss1 << "/home/manish/tcs/arc_setup/seg_ws/images/test1/" << it << ".jpg" ;




        //        imshow("im3", im3);
        //        cv::imwrite(ss1.str(),im3);
        //        cv::namedWindow("output_color_image",CV_WINDOW_NORMAL);


        cv::resize(color_output_image,color_output_image,cv::Size(input_image_width,input_image_height));
        cv::resize(input_image,input_image,cv::Size(input_image_width,input_image_height));
        cv::resize(mask,mask,cv::Size(input_image_width,input_image_height));
        imshow("output_color_image",color_output_image);
        imshow("input_image",input_image);
        imshow("mask_genrated",mask);

        stringstream ee,ee1,ee2;

        ee << "/home/manish/Desktop/" << it << ".png";
        ee2 << "/home/manish/Desktop/image_" << it << ".png";
        ee1 << "/home/manish/Desktop/mask_" << it << ".png";



        cvtColor(mask,mask,CV_GRAY2BGR);

        Size sz1 = input_image.size();
        Size sz2 = color_output_image.size();
        Mat im3(sz1.height, sz1.width+sz2.width+sz2.width, CV_8UC3);
        input_image.copyTo(im3(Rect(0, 0, sz1.width, sz1.height)));
        color_output_image.copyTo(im3(Rect(sz1.width, 0, sz2.width, sz2.height)));
        mask.copyTo(im3(Rect(sz1.width+sz1.width, 0, sz2.width, sz2.height)));

        imwrite(ee.str(),color_output_image);
        //       imwrite(ee1.str(),mask);
        imwrite(ee2.str(),im3);


        ee2.str("");
        ee.str("");
        ee1.str("");

        waitKey(0);
        //        it+=160;
        it+=3;
    }

    return 0;
}

