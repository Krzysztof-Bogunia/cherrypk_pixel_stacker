// #include <iostream>
// #include <fstream>
// #include <string>
// #include <opencv2/core.hpp>
// // #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include "opencv2/highgui.hpp"
// #include "opencv2/core/base.hpp"
// #include "opencv2/core/mat.hpp"
// #include "opencv2/features2d.hpp"
// #include <chrono>
// #include <vector>
// #include <list>
// #include <algorithm>
// #include <glob.h>
// #include <filesystem>
// #include <cmath>
// #include <valarray>
// #include <random>
// #include <list>
// #include <filesystem>
// #include <ctime>
// #include "opencv2/photo.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.cpp"
#include "processing.cpp"
#include "colorlut.cpp"


int main(int argc, char* argv[])
{
  using namespace std;
  using namespace cv;
  
  auto start = std::chrono::high_resolution_clock::now();

  VERBOSITY = 0; //increases number of information that program shows. May result in creation of additional files. //0

  // cv::theRNG().state = 1;

  string SettingsPath = "./settings.json";
  string AlignedPath = "./aligned/";
  bool mask_set_manual = false;
  string MasksPath = "./masks/";
  bool task_set_manual = false;
  ProgramTask Task = ProgramTask::stack; //ProgramTask::stack; ProgramTask::calibrate_color;
  string ImagesPath = "./input/";
  string TargetImagesPath = "./target/"; // "./reference/"  ""
  bool output_set_manual = false;
  string OutputPath = "./output/";
  if (argc >= 2) {
    for (int i = 0; i < argc; ++i) {
      if(string(argv[i]) == "--settings") {
        SettingsPath = argv[i+1];
        i++;
      }
      else if(string(argv[i]) == "--aligned") {
        AlignedPath = argv[i+1];
        i++;
      }
      else if(string(argv[i]) == "--masks") {
        MasksPath = argv[i+1];
        i++;
      }
      else if(string(argv[i]) == "--images") {
        ImagesPath = argv[i+1];
        i++;
      }
      else if(string(argv[i]) == "--output") {
        OutputPath = argv[i+1];
        output_set_manual = true;
        i++;
      }
      else if(string(argv[i]) == "--task") {
        Task = toProgramTask(string(argv[i+1]));
        task_set_manual = true;
        i++;
      }
      else if(string(argv[i]) == "--target") {
        TargetImagesPath = argv[i+1];
        i++;
      }
    }
  }
  // SettingsPath = SettingsPath + "/";
  AlignedPath = AlignedPath + "/";
  MasksPath = MasksPath + "/";
  ImagesPath = ImagesPath + "/";
  OutputPath = OutputPath + "/";
  TargetImagesPath = TargetImagesPath + "/";
  SettingsPath = ReplaceAll(SettingsPath, "//", "/");
  AlignedPath = ReplaceAll(AlignedPath, "//", "/");
  MasksPath = ReplaceAll(MasksPath, "//", "/");
  ImagesPath = ReplaceAll(ImagesPath, "//", "/");
  OutputPath = ReplaceAll(OutputPath, "//", "/");
  TargetImagesPath = ReplaceAll(TargetImagesPath, "//", "/");
  std::string ImgsFullPath = ImagesPath + "*.jpg";
  ImgsFullPath = ReplaceAll(ImgsFullPath, "//", "/");
  std::string TargetFullPath = TargetImagesPath + "*.jpg";

  // LOAD DATA  
  auto[programParams1, alignmentParams1, stackingParams1, colorParams1] = LoadProgramParameters(SettingsPath);
  //Program Params
  VERBOSITY = programParams1.VERBOSITY;
  if(!task_set_manual) {
    Task = programParams1.task;
  }
  auto interpolation = (cv::InterpolationFlags)programParams1.interpolation; //cv::INTER_LANCZOS4; //cv::INTER_LINEAR; cv::INTER_LANCZOS4
  float radicalChangeRatio = programParams1.radicalChangeRatio;
  //Alignment Params
  int base_index = alignmentParams1.base_index; //index of base reference image //0
  double checkArea = alignmentParams1.checkArea; //0.7 //0.8 0.9
  double alpha = alignmentParams1.alpha; //0.7; //1.0
  int maxIter = alignmentParams1.maxIter; //100; //1 // 50; 15
  bool alignCenter = alignmentParams1.alignCenter; //false; //false
  bool warpAlign = alignmentParams1.warpAlign; //true;
  int warpIter = alignmentParams1.warpIter; //0
  int K = alignmentParams1.K; //3; //matchedPoints2.rows/7; //3
  int n_points = alignmentParams1.n_points; //1024; //4096 //8192 //512
  float ratio = alignmentParams1.ratio; //0.75f; //how many points to keep for alignment //0.5f;
  bool mirroring = alignmentParams1.mirroring; //false; //try mirroring best alignment //false
  int erosion_size = alignmentParams1.erosion_size; //3; //cut borders of mask
  //Stacking Params
  int patternN = stackingParams1.patternN; //200; //64
  int patternSize = stackingParams1.patternSize; //3; //16
  float minImgCoef = stackingParams1.minImgCoef; //0.0f;
  float baseImgCoef = stackingParams1.baseImgCoef; //0.5f; //0.5f;
  float coef_sharpness = stackingParams1.coef_sharpness; //1.0; //3.0;
  float coef_similarity = stackingParams1.coef_similarity; //1.0;
  double comparison_scale = stackingParams1.comparison_scale; //1.0; //0.5
  //Color Params
  int num_dominant_colors = colorParams1.num_dominant_colors; //how many colors to use for alignment 3
  int histSize = colorParams1.histSize; // //256 //32 //64                     
  float colorStrength = colorParams1.strength; //how much to change/align the color //0.5f 1.0f
  float maxChange = colorParams1.maxChange; //limit ratio (original*(1+maxChange)) for max color change //0.5f
  int find_colors = colorParams1.find_colors; //how many colors to search for best match //num_dominant_colors+2
  //LOAD IMAGES
  auto[images, imagesPaths] = LoadImages(ImgsFullPath, -1, true);
  int N = images.size();
  std::cout << "Number of loaded photos from path [" + ImgsFullPath + "] : " << N << std::endl;
  // std::reverse(images.begin(), images.end()); //images in order from far focus to close focus
  // std::reverse(imagesPaths.begin(), imagesPaths.end());
  std::vector<std::string> imagesNames;
  for (int i = 0; i < N; i++)
  {
    imagesNames.push_back(std::filesystem::path(imagesPaths[i]).filename().string());
      // std::cout << "- " + imagesNames[i] << std::endl;
  }
  vector<Mat> targetImages;
  vector<string> targetImagesPaths, targetImagesNames;
  if(Task == ProgramTask::calibrate_color) {
    auto[targetImages_, targetImagesPaths_] = LoadImages(TargetFullPath, -1, true);
    targetImages = targetImages_;
    targetImagesPaths = targetImagesPaths_;
    std::cout << "Number of loaded target photos from path [" + TargetFullPath + "] : " << targetImages.size() << std::endl;
    for (int i = 0; i < targetImages.size(); i++)
    {
      targetImagesNames.push_back(std::filesystem::path(targetImagesPaths[i]).filename().string());
      // std::cout << "- " + targetImagesNames[i] << std::endl;
    }
  }

  time_t timestamp = time(NULL);
  struct tm datetime = *localtime(&timestamp);
  char dateFormatted[16];
  strftime(dateFormatted, 16, "%Y%m%d_%H%M%S", &datetime);

  //set base/reference image
  Mat baseImg = images[base_index].clone();
  int w = baseImg.cols;
  int h = baseImg.rows;
  auto newCameraResolution = baseImg.size();
  int cameraWidth = newCameraResolution.width;
  int cameraHeight = newCameraResolution.height;

  //match resolution to base image
  if((Task == ProgramTask::stack) || (Task == ProgramTask::simple_stack) || (Task == ProgramTask::align)) {
    vector<cv::Mat> images2(images.size());
    #pragma omp parallel for
    for (int i = 0; i < N; i++)
    {
      // test: images in lower resolution
      // images2.push_back(ReduceResolution(images[i], 1000, cv::INTER_LANCZOS4));
      images2[i] = Resize(images[i], w, h, interpolation);
    }
    images = images2;
    images2.clear();
  }

  vector<Mat> imagesForProcessing(images.size());
  vector<Mat> masks(images.size());
  vector<double> mseList_unprocessed(images.size());
  vector<double> mseList(images.size());
  Mat baseImgMask = Mat::ones(baseImg.rows, baseImg.cols, CV_32F);
  for (int i = 0; i < N; i++) {
    imagesForProcessing[i] = images[i];
    Mat ImgMask = Mat::ones(imagesForProcessing[i].rows, imagesForProcessing[i].cols, CV_32F);
    masks[i] = ImgMask;
  }
  imagesForProcessing[base_index] = baseImg;


  if((Task == ProgramTask::stack) || (Task == ProgramTask::align)) {
    // Mat baseImgMask(cv::Size(baseImg.cols, baseImg.rows), CV_32F, Scalar(1.0f)); 
    baseImgMask = Mat::ones(cv::Size(baseImg.cols, baseImg.rows), CV_32F); 
    masks[base_index] = baseImgMask;
    mseList[base_index] = CompareImg(baseImg, imagesForProcessing[base_index], 0.55);
    //get mse to compare before-after
    #pragma omp parallel for
    for (int i = 0; i < images.size(); i++) {
      mseList_unprocessed[i] = CompareImg(baseImg, images[i], 0.7);  
    }
  }

  // Preprocessing images for stacking (align/undistort)
  if((Task == ProgramTask::stack) || (Task == ProgramTask::align)) {
    //TODO: improve masking (avoid masking regions that were black in original images ?)
    #pragma omp parallel for
    for (int i = 0; i < images.size(); i++)
    {
      if(i != base_index) {
        auto [autoUndistorted, mask] = relativeUndistort(baseImg, images[i].clone(),cameraWidth, cameraHeight, programParams1, alignmentParams1);
        // mask.convertTo(mask, CV_32F, (double)1.0/255.0); //convert to float[0.0f, 1.0f]
        //mask bad/unknown pixels
        if(erosion_size > 0) {
          Mat element = getStructuringElement(MorphShapes::MORPH_RECT,
                                    Size(2*erosion_size+1, 2*erosion_size+1),
                                    Point(erosion_size, erosion_size));
          erode(mask, mask, element);
        }    
        Mat mask2 = maskFromChange(baseImg, autoUndistorted, radicalChangeRatio);
        mask = multiply(mask, mask2);
        imagesForProcessing[i] = autoUndistorted;
        masks[i] = mask;
        mseList[i] = CompareImg(baseImg, autoUndistorted, 0.7);  
      }
    }
    if(VERBOSITY >= 0) { 
      cout<<"Input images had misalignment error before prosessing: mse = "<<mseList_unprocessed<<endl; 
      cout<<"Images for stacking were undistorted with result: mse = "<<mseList<<endl; 
    }
  }


  //upscale images
  int newWidth, newHeight;
  if((Task == ProgramTask::stack) || (Task == ProgramTask::align)) {
    newWidth = (int)(stackingParams1.upscale * (double)baseImg.cols);
    newHeight = (int)(stackingParams1.upscale * (double)baseImg.rows);
    if(stackingParams1.upscale > 1.0) {
      // #pragma omp parallel for
      for (int i = 0; i < imagesForProcessing.size(); i++) {
        imagesForProcessing[i] = Resize(imagesForProcessing[i], newWidth, newHeight, interpolation);
        masks[i] = Resize(masks[i], newWidth, newHeight, INTER_NEAREST);
      }
    }
  }

  //realign after upscaling
  Mat baseImg_resized;
  if((Task == ProgramTask::stack) || (Task == ProgramTask::align)) {
    bool realigned2 = false;
    baseImg_resized = Resize(baseImg, newWidth, newHeight, interpolation);
    #pragma omp parallel for
    for (int i = 0; i < imagesForProcessing.size(); i++) {
      if((stackingParams1.upscale > 1.0) && (i != base_index)) {
        Mat autoAligned = AlignImageToImage(baseImg_resized, imagesForProcessing[i]);
        // Mat mask = maskFromColor(autoAligned, {0,0,0});
        Mat mask = add(maskFromColor(autoAligned), invertMask(maskFromColor(baseImg_resized)));
        cv::threshold(mask,	mask, 0.0, 1.0, THRESH_BINARY);
        mask.convertTo(mask, CV_32F);
        //mask bad/unknown pixels
        if(erosion_size > 0) {
          Mat element = getStructuringElement(MorphShapes::MORPH_ELLIPSE,
                                    Size(2*erosion_size+1, 2*erosion_size+1),
                                    Point(erosion_size, erosion_size));
          erode(mask, mask, element);
        }
        Mat mask2 = maskFromChange(baseImg_resized, autoAligned, radicalChangeRatio);
            // mask.convertTo(mask, CV_8U, (double)255.0);
            // mask2.convertTo(mask2, CV_8U, (double)255.0);
            // cv::imwrite(string("maskFromColor")+".jpg", mask);
            // cv::imwrite(string("maskFromChange")+".jpg", mask2);
        mask = multiply(mask, mask2);
        imagesForProcessing[i] = autoAligned;
        masks[i] = mask;
        // mseList[i] = CompareImg(baseImg_resized, autoAligned, 0.7);
        mseList[i] = CompareImg(multiply(baseImg_resized, mask), multiply(autoAligned, mask), 0.7);
        realigned2 = true;
      }
    }
    if((VERBOSITY >= 0) && realigned2) { 
      cout<<"Images for stacking were re-aligned after upscaling with result: mse of valid(masked) regions = "<<mseList<<endl; 
    }
  }

  //retry alignment
  if((Task == ProgramTask::stack) || (Task == ProgramTask::align)) {
    bool realigned = false;
    // #pragma omp parallel for
    for (int i = 0; i < imagesForProcessing.size(); i++) {
      if(i != base_index) {
      // if(mseList[i] > mseList_unprocessed[i]) {
        if((alignmentParams1.splitAlignPartsHorizontal > 0) && (alignmentParams1.splitAlignPartsVertical > 0) && (i != base_index)) {
          // Mat autoAligned = AlignImageToImage(baseImg, images[i].clone());
          Mat autoAligned = AlignImageToImageRegions(baseImg_resized, imagesForProcessing[i], 
                                                      Size2i(alignmentParams1.splitAlignPartsHorizontal, alignmentParams1.splitAlignPartsVertical), 
                                                      nullptr, interpolation);
          // Mat mask = maskFromColor(autoAligned, {0,0,0});
          Mat mask = add(maskFromColor(autoAligned), invertMask(maskFromColor(baseImg_resized)));
          cv::threshold(mask,	mask, 0.0, 1.0, THRESH_BINARY);
          mask.convertTo(mask, CV_32F);
          //mask bad/unknown pixels
          if(erosion_size > 0) {
            Mat element = getStructuringElement(MorphShapes::MORPH_ELLIPSE,
                                      Size(2*erosion_size+1, 2*erosion_size+1),
                                      Point(erosion_size, erosion_size));
            erode(mask, mask, element);
          }
          Mat mask2 = maskFromChange(baseImg_resized, autoAligned, radicalChangeRatio);
              // mask.convertTo(mask, CV_8U, (double)255.0);
              // mask2.convertTo(mask2, CV_8U, (double)255.0);
              // cv::imwrite(string("maskFromColor")+".jpg", mask);
              // cv::imwrite(string("maskFromChange")+".jpg", mask2);
          mask = multiply(mask, mask2);
          imagesForProcessing[i] = autoAligned;
          masks[i] = mask;
          // mseList[i] = CompareImg(baseImg, autoAligned, 0.7);
          mseList[i] = CompareImg(multiply(baseImg_resized, mask), multiply(autoAligned, mask), 0.7);
          realigned = true;
        }
      }
    }
    if((VERBOSITY >= 0) && realigned) { 
      cout<<"Images for stacking were split and re-aligned with result: mse of valid(masked) regions = "<<mseList<<endl; 
    }
  }

   if(Task == ProgramTask::calibrate_color) {
    ColorLUT mylut;
    mylut.num_dominant_colors = num_dominant_colors;
    mylut.histSize = histSize;
    mylut.strength = colorStrength;
    mylut.maxChange = maxChange;
    mylut.find_colors = find_colors;
    // mylut.calibrate(imagesForProcessing, targetImages);
    vector<Mat> optimizedImages1(imagesForProcessing.size());
    vector<Mat> optimizedImages2(targetImages.size());
    #pragma omp parallel for
    for (int i = 0; i < imagesForProcessing.size(); i++)
    {
      auto img = ReduceResolution(imagesForProcessing[i], 2000, cv::INTER_NEAREST);
      cvtColor(img, img, COLOR_BGR2RGB);
      optimizedImages1[i] = img;

    }
    #pragma omp parallel for
    for (int i = 0; i < targetImages.size(); i++)
    {
      auto img = ReduceResolution(targetImages[i], 2000, cv::INTER_NEAREST);
      cvtColor(img, img, COLOR_BGR2RGB);
      optimizedImages2[i] = img;
    }
    mylut.calibrate(optimizedImages1, optimizedImages2);
    mylut.save("lut", true, false);
    mylut.save1d("lut1d", true, false);

    // Check if results folders exist
    namespace fs = std::filesystem;
    if (!fs::is_directory(OutputPath) || !fs::exists(OutputPath)) { 
          fs::create_directory(OutputPath);
    }
    for(int n1=0; n1<imagesForProcessing.size(); n1++) {
      Mat img = imagesForProcessing[n1];
      cvtColor(img, img, COLOR_BGR2RGB);
      Mat colorCalibrated = mylut.apply(img);
      // Mat colorCalibrated2 = mylut.apply2(imagesForProcessing[n1]);
      // SaveImage(colorCalibrated2, OutputPath+imagesNames[n1].substr(0, imagesNames[n1].find("."))+"_colorCalibrated_", ".jpg", false, true);
      SaveImage(colorCalibrated, OutputPath+imagesNames[n1].substr(0, imagesNames[n1].find("."))+"_colorCalibrated_", ".jpg",
                 true, true, true);
    }
  }
  // save intermediate preprocessed images
  if((Task == ProgramTask::stack) || (Task == ProgramTask::align)) {
    for (int i = 0; i < imagesForProcessing.size(); i++)
    {
      // Check if results folders exist
      namespace fs = std::filesystem;
      if((Task == ProgramTask::stack) || (Task == ProgramTask::align)) {
        string SavePath = AlignedPath;
        if(output_set_manual) {
          SavePath = OutputPath;
        }
        if (!fs::is_directory(SavePath) || !fs::exists(SavePath)) { 
            fs::create_directory(SavePath);
        }
        auto aligned = imagesForProcessing[i];
        cv::imwrite(SavePath+imagesNames[i].substr(0, imagesNames[i].find("."))+"_aligned_"+dateFormatted+".jpg", aligned);

        SavePath = MasksPath;
        if(output_set_manual) {
          SavePath = OutputPath;
        }
        if(mask_set_manual) {
          SavePath = MasksPath;
        }
        if (!fs::is_directory(SavePath) || !fs::exists(SavePath)) { 
            fs::create_directory(SavePath);
        }
        auto mask = masks[i];
        mask.convertTo(mask, CV_8U, (double)255.0);
        cv::imwrite(SavePath+imagesNames[i].substr(0, imagesNames[i].find("."))+"_mask_"+dateFormatted+".jpg", mask);
      }
      // else if(Task == ProgramTask::calibrate_color) {
      //   if (!fs::is_directory(OutputPath) || !fs::exists(OutputPath)) { 
      //       fs::create_directory(OutputPath);
      //   }
      //   auto colorCalibrated = imagesForProcessing[i];
      //   cv::imwrite(OutputPath+imagesNames[i].substr(0, imagesNames[i].find("."))+"_colorCalibrated_"+dateFormatted+".jpg", colorCalibrated);
      // }
    }
  }


  // stack images with maximizing sharpness
  if((Task == ProgramTask::stack) || (Task == ProgramTask::simple_stack)) {
    auto stackedImage = stackImages(imagesForProcessing, base_index, masks, programParams1, stackingParams1);
    // if(stackingParams1.upscale > 1.0) {
    //   int newWidth = (int)(stackingParams1.upscale * (double)baseImg.cols);
    //   int newHeight = (int)(stackingParams1.upscale * (double)baseImg.rows);
    //   stackedImage = Resize(stackedImage, newWidth, newHeight, interpolation);
    // }


    auto[errorMSE_stacked, psnr_stacked] = CompareMetrics(imagesForProcessing[base_index], stackedImage, true);
    auto sharpnessRefImg = sharpnessOfRegions(imagesForProcessing[base_index], 4, 64);
    auto sharpnessPerRegion = sharpnessOfRegions(MatchResolution(stackedImage, imagesForProcessing[base_index].size(), INTER_NEAREST), 4, 64);
    valarray<float> sharpness = toValarray(sharpnessRefImg);
    float meanSharp = sharpness.sum() / (sharpness.size());
    valarray<float> sharpness2 = toValarray(sharpnessPerRegion);
    float meanSharp2 = sharpness2.sum() / (sharpness2.size());
    if(VERBOSITY >= 0) {
      cout<<"Average sharpness of base image: "<< meanSharp<<endl;
      cout<<"Average sharpness of stacked image: "<< meanSharp2<<endl;
    }
    Mat resultImage;
    // stackedImage.convertTo(resultImage, CV_32F, 1.0/255.0);
    stackedImage.convertTo(resultImage, CV_8U);
    SaveImage(resultImage, OutputPath+"stacked_"+dateFormatted, ".png", true);
  }

  if(VERBOSITY >= 0) {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "total elapsed time = "<< MicroToSeconds(duration.count()) << " [s]" << std::endl;  
  }
  cout << "All processing done. Program exit." << endl;
  return 0;
}
