#include <iostream>
#include <fstream>
#include <string>
#include <opencv2/core.hpp>
// #include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
// #include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include <chrono>
#include <vector>
#include <list>
#include <algorithm>
#include <glob.h>
#include <filesystem>
#include <cmath>
#include <valarray>
#include <random>
#include <list>
#include <filesystem>
#include <ctime>
#include "processing.cpp"



int main(int argc, char* argv[])
{
  using namespace std;
  using namespace cv;
  
  auto start = std::chrono::high_resolution_clock::now();

  VERBOSITY = 0; //increases number of information that program shows. May result in creation of additional files. //0

  // cv::theRNG().state = 1;

  string SettingsPath = "./settings.json";
  string AlignedPath = "./aligned/";
  string MasksPath = "./masks/";
  std::string ImagesPath = "./input/";
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
    }
  }
  // SettingsPath = SettingsPath + "/";
  AlignedPath = AlignedPath + "/";
  MasksPath = MasksPath + "/";
  std::string imgsFullPath = ImagesPath + "/" + "*.jpg";
  SettingsPath = ReplaceAll(SettingsPath, "//", "/");
  AlignedPath = ReplaceAll(AlignedPath, "//", "/");
  MasksPath = ReplaceAll(MasksPath, "//", "/");
  imgsFullPath = ReplaceAll(imgsFullPath, "//", "/");

  // Load data  
  auto[programParams1, alignmentParams1, stackingParams1] = LoadProgramParameters(SettingsPath);

  auto[images, imagesPaths] = LoadImages(imgsFullPath, -1, true);
  int N = images.size();
  std::cout << "Number of loaded photos from path [" + imgsFullPath + "] : " << N << std::endl;
  // std::reverse(images.begin(), images.end()); //images in order from far focus to close focus
  // std::reverse(imagesPaths.begin(), imagesPaths.end());

  std::vector<std::string> imagesNames;
  for (int i = 0; i < N; i++)
  {
    imagesNames.push_back(std::filesystem::path(imagesPaths[i]).filename().string());
    std::cout << "- " + imagesNames[i] << std::endl;
  }

  auto interpolation = (cv::InterpolationFlags)programParams1.interpolation; //cv::INTER_LANCZOS4; //cv::INTER_LINEAR; cv::INTER_LANCZOS4
  float radicalChangeRatio = programParams1.radicalChangeRatio;

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

  int patternN = stackingParams1.patternN; //200; //64
  int patternSize = stackingParams1.patternSize; //3; //16
  float minImgCoef = stackingParams1.minImgCoef; //0.0f;
  float baseImgCoef = stackingParams1.baseImgCoef; //0.5f; //0.5f;
  float coef_sharpness = stackingParams1.coef_sharpness; //1.0; //3.0;
  float coef_similarity = stackingParams1.coef_similarity; //1.0;
  double comparison_scale = stackingParams1.comparison_scale; //1.0; //0.5

  Mat baseImg = images[base_index].clone();
  int w = baseImg.cols;
  int h = baseImg.rows;
  auto newCameraResolution = baseImg.size();
  int cameraWidth = newCameraResolution.width;
  int cameraHeight = newCameraResolution.height;

  //match resolution to base image
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

  vector<Mat> imagesForStack(images.size());
  imagesForStack[base_index] = baseImg;
  // if(VERBOSITY > 1) {
  //   cv::imwrite("./undistortedGPU_baseImg" + string(".jpg"), baseImg); //test
  // }
  vector<Mat> masks(images.size());
  vector<double> mseList_unprocessed(images.size());
  vector<double> mseList(images.size());
  // Mat baseImgMask(cv::Size(baseImg.cols, baseImg.rows), CV_32F, Scalar(1.0f)); 
  Mat baseImgMask = Mat::ones(cv::Size(baseImg.cols, baseImg.rows), CV_32F); 
  masks[base_index] = baseImgMask;
  mseList[base_index] = CompareImg(baseImg, imagesForStack[base_index], 0.55);
  //get mse to compare before-after
  #pragma omp parallel for
  for (int i = 0; i < images.size(); i++) {
    mseList_unprocessed[i] = CompareImg(baseImg, images[i], 0.7);  
  }

  // Preprocessing images for stacking (align/undistort)
  //TODO: improve masking (avoid masking regions that were black in oryginal images ?)
  #pragma omp parallel for
  for (int i = 0; i < images.size(); i++)
  {
    if(i != base_index) {
      // // t1 = time.time()
      auto [autoUndistorted, mask] = RelativeUndistort(baseImg, images[i].clone(),cameraWidth, cameraHeight, programParams1, alignmentParams1);
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
      imagesForStack[i] = autoUndistorted;
      masks[i] = mask;
      mseList[i] = CompareImg(baseImg, autoUndistorted, 0.7);  
    }
  }
  if(VERBOSITY >= 0) { 
    cout<<"Input images had misalignment error before prosessing: mse = "<<mseList_unprocessed<<endl; 
    cout<<"Images for stacking were undistorted with result: mse = "<<mseList<<endl; 
  }


  //upscale images
  int newWidth = (int)(stackingParams1.upscale * (double)baseImg.cols);
  int newHeight = (int)(stackingParams1.upscale * (double)baseImg.rows);
  if(stackingParams1.upscale > 1.0) {
    // #pragma omp parallel for
    for (int i = 0; i < imagesForStack.size(); i++) {
      imagesForStack[i] = Resize(imagesForStack[i], newWidth, newHeight, interpolation);
      masks[i] = Resize(masks[i], newWidth, newHeight, INTER_NEAREST);
    }
  }

  //realign after upscaling
  bool realigned2 = false;
  Mat baseImg_resized = Resize(baseImg, newWidth, newHeight, interpolation);
  #pragma omp parallel for
  for (int i = 0; i < imagesForStack.size(); i++) {
    if((stackingParams1.upscale > 1.0) && (i != base_index)) {
      Mat autoAligned = AlignImageToImage(baseImg_resized, imagesForStack[i]);
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
      imagesForStack[i] = autoAligned;
      masks[i] = mask;
      // mseList[i] = CompareImg(baseImg_resized, autoAligned, 0.7);
      mseList[i] = CompareImg(multiply(baseImg_resized, mask), multiply(autoAligned, mask), 0.7);
      realigned2 = true;
    }
  }
  if((VERBOSITY >= 0) && realigned2) { 
    cout<<"Images for stacking were re-aligned after upscaling with result: mse of valid(masked) regions = "<<mseList<<endl; 
  }

  //retry alignment
  bool realigned = false;
  // #pragma omp parallel for
  for (int i = 0; i < imagesForStack.size(); i++) {
    if(i != base_index) {
    // if(mseList[i] > mseList_unprocessed[i]) {
      if((alignmentParams1.splitAlignPartsHorizontal > 0) && (alignmentParams1.splitAlignPartsVertical > 0) && (i != base_index)) {
        // Mat autoAligned = AlignImageToImage(baseImg, images[i].clone());
        Mat autoAligned = AlignImageToImageRegions(baseImg_resized, imagesForStack[i], 
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
        imagesForStack[i] = autoAligned;
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

  // save intermediate preprocessed images
  time_t timestamp = time(NULL);
  struct tm datetime = *localtime(&timestamp);
  char dateFormatted[16];
  strftime(dateFormatted, 16, "%Y%m%d_%H%M%S", &datetime);
  for (int i = 0; i < imagesForStack.size(); i++)
  {
    // Check if results folders exist
    namespace fs = std::filesystem;
    if (!fs::is_directory(AlignedPath) || !fs::exists(AlignedPath)) { 
        fs::create_directory(AlignedPath);
    }
    if (!fs::is_directory(MasksPath) || !fs::exists(MasksPath)) { 
        fs::create_directory(MasksPath);
    }
    auto aligned = imagesForStack[i];
    auto mask = masks[i];
    mask.convertTo(mask, CV_8U, (double)255.0);
    cv::imwrite(AlignedPath+imagesNames[i].substr(0, imagesNames[i].find("."))+"_aligned_"+dateFormatted+".jpg", aligned);
    cv::imwrite(MasksPath+imagesNames[i].substr(0, imagesNames[i].find("."))+"_mask_"+dateFormatted+".jpg", mask);
  }


  // stack images with maximizing sharpness
  auto stackedImage = StackImages(imagesForStack, base_index, masks, programParams1, stackingParams1);
  // if(stackingParams1.upscale > 1.0) {
  //   int newWidth = (int)(stackingParams1.upscale * (double)baseImg.cols);
  //   int newHeight = (int)(stackingParams1.upscale * (double)baseImg.rows);
  //   stackedImage = Resize(stackedImage, newWidth, newHeight, interpolation);
  // }

  auto[errorMSE_stacked, psnr_stacked] = CompareMetrics(imagesForStack[base_index], stackedImage, true);
  auto sharpnessRefImg = SharpnessOfRegions(imagesForStack[base_index], 4, 64);
  auto sharpnesPerRegion = SharpnessOfRegions(MatchResolution(stackedImage, imagesForStack[base_index].size(), INTER_NEAREST), 4, 64);
  valarray<float> sharpness = toValarray(sharpnessRefImg);
  float meanSharp = sharpness.sum() / (sharpness.size());
  valarray<float> sharpness2 = toValarray(sharpnesPerRegion);
  float meanSharp2 = sharpness2.sum() / (sharpness2.size());
  if(VERBOSITY >= 0) {
    cout<<"Average sharpness of base image: "<< meanSharp<<endl;
    cout<<"Average sharpness of stacked image: "<< meanSharp2<<endl;
  }
  Mat resultImage;
  // stackedImage.convertTo(resultImage, CV_32F, 1.0/255.0);
  stackedImage.convertTo(resultImage, CV_8U);
  // SaveImages([stackedImage], "./stacked_images", suffix=sceneName+"_stacked"+str(len(imagesForStack))
  //                                                                     +"_Psnr_"+str(round(psnr_stacked,2))
  //                                                                     +"_sharp_"+str(round(np.mean(sharpnesPerRegion),2))
  //                                                                     + "_patN_"+str(patternN)
  //                                                                     +"_patSize_"+str(patternSize)
  //                                                                     +"_minCoef_"+str(minImgCoef)
  //                                                                     +"_baseImgCoef_"+str(baseImgCoef)
  //                                                                     +"_checkArea_"+str(checkArea)
  //                                                                     +"_alpha_"+str(alpha)
  //                                                                     +"_interp_"+str(interpolation)
  //                                                                     +"_maxIter_"+str(maxIter)
  //                                                                     +"_betterSharp_"+str(round(betterSharp,3))
  //                                                                     +"_eccIter_"+str(eccIter)
  //                                                                     +"_ssim_"+str(round(ssim_stacked,5)),
  //                                                                     imgFormat=".png" )
  SaveImage(resultImage, "stacked", ".png", true);

  if(VERBOSITY >= 0) {
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    std::cout << "total elapsed time = "<< MicroToSeconds(duration.count()) << " [s]" << std::endl;  
  }
  cout << "All processing done. Program exit." << endl;
  return 0;
}
