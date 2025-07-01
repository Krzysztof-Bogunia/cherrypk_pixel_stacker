#pragma once

// #include <iostream>
// #include <fstream>
// #include <string>
// #include <opencv2/core.hpp>
// // #include <opencv2/opencv.hpp>
// #include <opencv2/imgproc.hpp>
// #include "opencv2/highgui.hpp"
// #include "opencv2/features2d.hpp"
// #include "opencv2/video.hpp"
// #include "opencv2/calib3d.hpp"
// #include <chrono>
// #include <vector>
// #include <list>
// #include <algorithm>
// #include <glob.h>
// #include <filesystem>
// #include <cmath>
// #include <valarray>
// #include <random>
// #include <limits>
// #include <unordered_set>
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/traits.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.cpp"
#include <libInterpolate/Interpolate.hpp>
#include <string>
#include <valarray>
#include <vector>

enum Detector {
  orb,
  kaze,
  sift,
  akaze
};

enum ProgramTask {
  stack,
  simple_stack,
  align,
  calibrate_color
};

struct ProgramParams {
  int VERBOSITY = 0;
  ProgramTask task = ProgramTask::stack;
  float radicalChangeRatio = 1.3f;
  int interpolation = cv::INTER_LANCZOS4; //which interpolation algorithm to use (1:Linear, 4:Lanchos) //4 //cv::INTER_LINEAR; cv::INTER_LANCZOS4
  int erosion_size = 3; //expand mask of bad pixels //3
};
int VERBOSITY = 0; //0

struct AlignmentParams {
  int base_index = 0; //index of base reference image //0
  double checkArea = 0.85; //image comparison area //0.7
  double alpha = 0.9; //how many points to keep for alignment //1.0
  int maxIter = 30; //max number of undistortion iterations //50
  bool alignCenter = false; //keep center of images the same //false
  bool warpAlign = true; //apply warp perspective operation to align images //true
  int splitAlignPartsVertical = 8; //how many times to split image (vertically) to align each part independently //4
  int splitAlignPartsHorizontal = 8; //how many times to split image (horizontally) to align each part independently //4
  int warpIter = 40; //max number of align image iterations //0
  int n_points = 8000; //initial number of points to detect and compare between images //1024
  int K = -1;  //number of points clusters for estimating distortion //3
  float ratio = 0.65f; //how many points to keep for undistortion //0.65f,
  bool mirroring = false; //try mirroring best alignment //false
  int erosion_size = 3; //cutting size in pixels for borders of mask //3
};

struct StackingParams {
  int patternN = 200; //number of sharpness checking regions (total=patternN*patternN) //200
  int patternSize = 5; //size of each sharpness checking region //3
  float minImgCoef = 0.0f; //minimum value to add to each image's coefficients //0.0
  float baseImgCoef = 0.4f; //coefficient value of base image (by default first img is base) //0.5f
  float coef_sharpness = 1.5; //local sharpness weight for total image coeffs //1.0
  float coef_similarity = 1.0; //local similarity to base img weight for total image coeffs //1.0
  double comparison_scale = 0.25; //pixel ratio - decrease resolution for calculating some parameters //1.0
  int blur_size = 5; //adds smoothing to coefficients (increase it to hide switching pixel regions between images)
  double upscale = 1.0; //if value is greater than 1.0 then final image will be upscaled (by upscaling input images and merging them) //1.0
};

struct ColorParams {
  int histSize = 65; //number or color values per channel //32 //64 
  int num_dominant_colors = 16; //how many colors to use for alignment 3
  int find_colors = 20; //how many colors to search for best match //num_dominant_colors*1.5
  float strength = 1.0f; //how much to change/align the color //1.0f
  float maxChange = 0.15; //limit ratio (original*(1+maxChange)) for max color change //0.1f
};


/**
 * @brief Return vector<int> indices that would sort input vector
 */
template <class T>
std::vector<int> Argsort(std::vector<T> vec)
{
  std::vector<int> indices(vec.size());
  std::iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(), [&](int i, int j) { return vec[i] < vec[j]; });
  return indices;
}

/**
 * @brief Return vector<int> indices that would sort input vector
 */
template <class T>
std::vector<int> Argsort(std::vector<std::vector<T>> vec2)
{
  std::vector<int> indices(vec2.size());
  std::iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(), [&](int i, int j) { return vec2[i][0] < vec2[j][0]; });
  return indices;
}

/**
 * @brief Return vector<int> indices that would sort input vector
 */
std::vector<int> Argsort(std::vector<std::string> vec)
{
  auto vec2 = vec;
  // for (int i = 0; i < vec2.size(); i++)
  // {
  //   // reverse(vec2[i].begin(), vec2[i].end());
  //   // vec2[i] = std::string(vec2[i].rbegin(), vec2[i].rend());
  //   vec2[i] = std::string(vec2[i].rbegin(), vec2[i].rend());
  // }
  std::vector<int> indices(vec2.size());
  std::iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(), [&](int i, int j) { return isGreater(vec2[j], vec2[i]); });
  // sort(indices.begin(), indices.end(), [&vec](std::string i, std::string j) { return (bool)i.compare(j); });
  return indices;
}

template <class T>
std::vector<T> Reorder(const std::vector<T> &v, const std::vector<int> &order)
{
  std::vector<T> vec2(v.size());
  for (int i = 0, d; i < order.size(); i++)
  {
    vec2[i] = v[order[i]];
  }
  return vec2;
}


/// Initialize 2D maps with values in range [0...n].
/// @returns Pair of 2D maps of float32 type. 1st for transposition of first dimension and 2nd for transposition of 2nd dim.
std::tuple<cv::Mat, cv::Mat> initMap2D(int rows, int cols)
{
  using namespace std;
  using namespace cv;

  Mat X = Mat::zeros(rows, cols, CV_32F);
  Mat Y = Mat::zeros(rows, cols, CV_32F);

  for (int n = 0; n < rows; n++)
  {
    for (int n2 = 0; n2 < cols; n2++)
    {
      X.at<float>(n, n2) = (float)n2; //X[n * size2 + n2] = (S)n;
      Y.at<float>(n, n2) = (float)n; //Y[n * size2 + n2] = (S)n2;
    }
  }

  return std::make_tuple(X, Y);
}

cv::Mat warpPerspective(const cv::Mat &image, const cv::Mat &H, const cv::Size &size, cv::InterpolationFlags interpolation=cv::INTER_LINEAR)
{
  using namespace std;
  using namespace cv;

  // Mat Hinv = H.inv();

  int width = size.width;
  int height = size.height;

  // // Init Maps
  // cv::Mat map1(image.size(), cv::DataType<float>::type);
  // cv::Mat map2(image.size(), cv::DataType<float>::type);
  // Mat pXi_objPoints;
  // Mat pYi_objPoints;
  // auto[pXi_objPoints,pYi_objPoints] = initMap2D(size2.rows, size2.cols);
  // Mat pXf_objPoints;
  // Mat pYf_objPoints;
  // pXi_objPoints.convertTo(pXf_objPoints, cv::DataType<float>::type);
  // pYi_objPoints.convertTo(pYf_objPoints, cv::DataType<float>::type);

  cv::Mat unwarped(image.size(), cv::DataType<float>::type);
  cv::warpPerspective(image, unwarped, H, cv::Size(width, height), interpolation, cv::BORDER_CONSTANT, 0);
  return unwarped;
}

cv::Mat Undistort(const cv::Mat &image, const cv::Mat &mtx, const cv::Mat &dist, int width, int height, cv::InterpolationFlags interpolation=cv::INTER_LINEAR,
                  double alpha = 1.0, bool AlignCenter = true)
{
  using namespace cv;
  cv::Mat unwarped = image.clone();
  cv::Size oryginalCameraResolution(width,height);
  // cv::Size oryginalCameraResolution(mtx.at<float>(0, 2)*2.0, mtx.at<float>(1, 2)*2.0);
  auto newCameraResolution = cv::Size(image.cols, image.rows); // cv::Size(width, height); //cv::Size(image.cols, image.rows); //oryginalCameraResolution;
  //TODO: improve getting newcameramtx
  cv::Mat newcameramtx = mtx;
  Mat identityMat = Mat::eye(3, 3, CV_32F);
  if(std::equal(mtx.begin<float>(), mtx.end<float>(), identityMat.begin<float>())) {
    newcameramtx = mtx;
  }
  else {
    newcameramtx = cv::getOptimalNewCameraMatrix(mtx, dist, oryginalCameraResolution, alpha, newCameraResolution);
    // newcameramtx.at<float>(0, 2) = width/2;
    // newcameramtx.at<float>(1, 2) = height/2;
    // newcameramtx.at<float>(0, 0) = (float)width; //(float)std::min(width,height);
    // newcameramtx.at<float>(1, 1) = (float)height; //(float)std::min(width,height);
    if(VERBOSITY > 0) {
      std::cout << "Undistort: Warning: Using nondefault matrix" << std::endl;
    }
  }

  if(AlignCenter) {
    //TODO: possible error
    cv::Mat H = cv::Mat::eye(3, 3, CV_32F);
    H.at<float>(0, 2) = (float)(newCameraResolution.width/2) - newcameramtx.at<float>(0,2);
    H.at<float>(1, 2) = (float)(newCameraResolution.height/2) - newcameramtx.at<float>(1,2);
    unwarped = warpPerspective(image, H, newCameraResolution, interpolation);
  }
  // auto [map1, map2] = initUndistortRectifyMap(newcameramtx, dist, newCameraResolution, AlignCenter, device);
  Mat map1,map2;
  Mat R;
  initUndistortRectifyMap(newcameramtx, dist, R, newcameramtx, unwarped.size(), CV_32FC1, map1, map2);
  cv::Mat undistorted;
  cv::remap(unwarped, undistorted, map1, map2, interpolation, cv::BORDER_CONSTANT, 0);
  // cv::undistort(unwarped, undistorted, mtx, dist, newcameramtx);
  
  return undistorted;
}

cv::Mat Undistort(const cv::Mat &image, const cv::Mat &map1, const cv::Mat &map2, cv::InterpolationFlags interpolation=cv::INTER_LINEAR)
{
  using namespace cv;

  cv::Mat unwarped = image.clone();
  cv::BorderTypes border = cv::BorderTypes::BORDER_CONSTANT;
  cv::remap(unwarped, unwarped, map1, map2, interpolation, border, 0);
  return unwarped;
}


//** general logic functions **

ProgramTask toProgramTask(const std::string &str) {
  using namespace std;

  ProgramTask task;
  if(isEqual(str, "stack", true)) {
    task = ProgramTask::stack;
  }
  else if(isEqual(str, "simple_stack", true)) {
    task = ProgramTask::simple_stack;
  }
  else if(isEqual(str, "align", true)) {
    task = ProgramTask::align;
  }
  else if(isEqual(str, "calibrate_color", true)) {
    task = ProgramTask::calibrate_color;
  }

  return task;
}

/// @brief replaces all occurrences of substring "from" with string "to" in input string "str"
std::string ReplaceAll(const std::string &str, const std::string &from, const std::string &to)
{
  int start_pos = 0;
  std::string result = str;
  while ((start_pos = result.find(from, start_pos)) != std::string::npos)
  {
    result.replace(start_pos, from.length(), to);
    start_pos += to.length();
  }
  return result;
}

//TODO: deprecated function?
std::tuple<cv::Mat, cv::Mat, int, int, float> LoadParameters(const std::string &calibrationPath)
{
  using namespace std;
  ifstream file(calibrationPath);
  Json::Value loadeddict;
  Json::Reader reader;

  reader.parse(file, loadeddict);

  vector<vector<float>> mtxloaded = toVec2D(loadeddict["camera_matrix"]);
  if (mtxloaded.size() == 0)
    cout << "Warning: LoadParameters: mtxloaded.size = " << mtxloaded.size() << endl;

  vector<vector<float>> distloaded = toVec2D(loadeddict["dist_coeff"]);
  int cameraHeight = loadeddict["cameraHeight"].as<int>();
  int cameraWidth = loadeddict["cameraWidth"].as<int>();
  float cameraFocus = loadeddict["cameraFocus"].as<float>();

  cv::Mat cameraMatrix = cv::Mat(toMat(mtxloaded)).reshape(0, 3); // Assuming camera_matrix is a 3x3 matrix
  cv::Mat distCoeffs = cv::Mat(toMat(distloaded)).reshape(1);     // Assuming dist_coeff is a 1xN matrix
  cout << "Loaded camera parameters from: " << calibrationPath << endl;

  return std::make_tuple(cameraMatrix, distCoeffs, cameraHeight, cameraWidth, cameraFocus);
}

//TODO: deprecated function?
std::tuple<std::vector<cv::Mat>, std::vector<cv::Mat>, int, int, std::vector<float>> LoadAllParameters(std::string folderPath_)
{
  // load camera parameters
  std::vector<cv::Mat> matrices;
  std::vector<cv::Mat> distortions;
  int cameraHeight = 0;
  int cameraWidth = 0;
  std::vector<float> cameraFocus;
  std::vector<std::string> filePaths;
  glob_t glob_result;
  std::string folderPath = ReplaceAll(folderPath_, "//", "/");
  glob(folderPath.c_str(), GLOB_TILDE, NULL, &glob_result);
  for (int i = 0; i < glob_result.gl_pathc; ++i)
  {
    filePaths.push_back(glob_result.gl_pathv[i]);
  }
  globfree(&glob_result);

  for (const std::string &calibrationPath : filePaths)
  {
    auto [mtx_, dist_, cameraHeight_, cameraWidth_, cameraFocus_] = LoadParameters(calibrationPath);
    matrices.push_back(mtx_);
    distortions.push_back(dist_);
    cameraFocus.push_back(cameraFocus_);
    cameraHeight = cameraHeight_;
    cameraWidth = cameraWidth_;
  }
  return std::make_tuple(matrices, distortions, cameraHeight, cameraWidth, cameraFocus);
}

std::tuple<ProgramParams, AlignmentParams, StackingParams, ColorParams> LoadProgramParameters(const std::string &path)
{
  using namespace std;
  using namespace cv;

  ProgramParams programPars1;
  AlignmentParams alignPars1;
  StackingParams stackPars1;
  ColorParams colorPars1;

  ifstream file(path);
  Json::Value loadeddict;
  Json::Reader reader;
  reader.parse(file, loadeddict);

  if(loadeddict.isMember("radicalChangeRatio")) {
    programPars1.radicalChangeRatio = loadeddict["radicalChangeRatio"].as<float>();
  }
  if(loadeddict.isMember("VERBOSITY")) {
    programPars1.VERBOSITY = loadeddict["VERBOSITY"].as<int>();
  }
  if(loadeddict.isMember("task")) {
    programPars1.task = toProgramTask(loadeddict["task"].as<string>());
  }
  if(loadeddict.isMember("interpolation")) {
    programPars1.interpolation = loadeddict["interpolation"].as<int>();
  }
  if(loadeddict.isMember("erosion_size")) {
    programPars1.erosion_size = loadeddict["erosion_size"].as<int>();
  }
  VERBOSITY = programPars1.VERBOSITY;

  if(loadeddict.isMember("base_index")) {
    alignPars1.base_index = loadeddict["base_index"].as<int>();
  }
  if(loadeddict.isMember("checkArea")) {
    alignPars1.checkArea = loadeddict["checkArea"].as<double>();
  }
  if(loadeddict.isMember("alpha")) {
    alignPars1.alpha = loadeddict["alpha"].as<double>();
  }
  if(loadeddict.isMember("maxIter")) {
    alignPars1.maxIter = loadeddict["maxIter"].as<int>();
  }
  if(loadeddict.isMember("alignCenter")) {
    alignPars1.alignCenter = loadeddict["alignCenter"].as<bool>();
  }
  if(loadeddict.isMember("warpAlign")) {
    alignPars1.warpAlign = loadeddict["warpAlign"].as<bool>();
  }
  if(loadeddict.isMember("warpIter")) {
    alignPars1.warpIter = loadeddict["warpIter"].as<int>();
  }
  if(loadeddict.isMember("splitAlignPartsVertical")) {
    alignPars1.splitAlignPartsVertical = loadeddict["splitAlignPartsVertical"].as<int>();
  }
  if(loadeddict.isMember("splitAlignPartsHorizontal")) {
    alignPars1.splitAlignPartsHorizontal = loadeddict["splitAlignPartsHorizontal"].as<int>();
  }
  if(loadeddict.isMember("K")) {
    alignPars1.K = loadeddict["K"].as<int>();
  }
  if(loadeddict.isMember("n_points")) {
    alignPars1.n_points = loadeddict["n_points"].as<int>();
  }
  if(loadeddict.isMember("ratio")) {
    alignPars1.ratio = loadeddict["ratio"].as<float>();
  }
  if(loadeddict.isMember("mirroring")) {
    alignPars1.mirroring = loadeddict["mirroring"].as<bool>();
  }

  if(loadeddict.isMember("patternN")) {
    stackPars1.patternN = loadeddict["patternN"].as<int>();
  }
  if(loadeddict.isMember("patternSize")) {
    stackPars1.patternSize = loadeddict["patternSize"].as<int>();
  }
  if(loadeddict.isMember("minImgCoef")) {
    stackPars1.minImgCoef = loadeddict["minImgCoef"].as<float>();
  }
  if(loadeddict.isMember("baseImgCoef")) {
    stackPars1.baseImgCoef = loadeddict["baseImgCoef"].as<float>();
  }
  if(loadeddict.isMember("coef_sharpness")) {
    stackPars1.coef_sharpness = loadeddict["coef_sharpness"].as<float>();
  }
  if(loadeddict.isMember("coef_similarity")) {
    stackPars1.coef_similarity = loadeddict["coef_similarity"].as<float>();
  }
  if(loadeddict.isMember("comparison_scale")) {
    stackPars1.comparison_scale = loadeddict["comparison_scale"].as<double>();
  }
  if(loadeddict.isMember("blur_size")) {
    stackPars1.blur_size = loadeddict["blur_size"].as<int>();
  }
  if(loadeddict.isMember("upscale")) {
    stackPars1.upscale = loadeddict["upscale"].as<double>();
  }

  if(loadeddict.isMember("num_dominant_colors")) {
    colorPars1.num_dominant_colors = loadeddict["num_dominant_colors"].as<int>();
  }
  if(loadeddict.isMember("histSize")) {
    colorPars1.histSize = loadeddict["histSize"].as<int>();
  } 
  if(loadeddict.isMember("strength")) {
    colorPars1.strength = loadeddict["strength"].as<float>();
  } 
  if(loadeddict.isMember("maxChange")) {
    colorPars1.maxChange = loadeddict["maxChange"].as<float>();
  } 
  if(loadeddict.isMember("find_colors")) {
    colorPars1.find_colors = loadeddict["find_colors"].as<int>();
    colorPars1.find_colors = min(colorPars1.find_colors, colorPars1.histSize);
  } 

  cout << "Loaded program parameters from: " << path << endl;

  return std::make_tuple(programPars1, alignPars1, stackPars1, colorPars1);
}

/// Loads calibration pairs of missaligned points[height,width].
/// @param calibrationPath path to json file containing data.
/// @returns 2 sets of points[height,width] and image parameters {Height, Width, Focus}
std::tuple<cv::Mat, cv::Mat, int, int, float> LoadPoints(const std::string &calibrationPath)
{
  using namespace std;
  ifstream file(calibrationPath);
  Json::Value loadeddict;
  Json::Reader reader;

  reader.parse(file, loadeddict);

  vector<vector<float>> image1_points_raw = toVec2D(loadeddict["image1_points"]);
  if (image1_points_raw.size() == 0)
    cout << "Warning: LoadPoints: image1_points.size = " << image1_points_raw.size() << endl;

  vector<vector<float>> image2_points_raw = toVec2D(loadeddict["image2_points"]);
  int cameraHeight = loadeddict["cameraHeight"].as<int>();
  int cameraWidth = loadeddict["cameraWidth"].as<int>();
  float cameraFocus = loadeddict["cameraFocus"].as<float>();

  cv::Mat image1_points = toMat(image1_points_raw);
  cv::Mat image2_points = toMat(image2_points_raw);
  cout << "LoadPoints: Loaded camera parameters from: " << calibrationPath << endl;

  return std::make_tuple(image1_points, image2_points, cameraHeight, cameraWidth, cameraFocus);
}

std::tuple<std::vector<cv::Mat>, std::vector<std::string>> LoadImages(std::string stacking_path, int matchFirstDimSize = -1, bool printing = true)
{
  std::vector<std::string> imagesPaths;
  std::vector<std::string> imagesNames;
  glob_t glob_result;
  std::string stacking_path_ = ReplaceAll(stacking_path, "//", "/");
  glob(stacking_path_.c_str(), GLOB_TILDE, NULL, &glob_result);
  
  for (int i = 0; i < glob_result.gl_pathc; ++i)
  {
    imagesPaths.push_back(glob_result.gl_pathv[i]);
    auto words = split(imagesPaths[i], '/');
    words = split(words[words.size()-1], '.');
    imagesNames.push_back(words[words.size()-2]);
  }
  globfree(&glob_result);

  auto sortedIndices = Argsort(imagesNames);
  imagesPaths = Reorder(imagesPaths, sortedIndices);
  std::vector<cv::Mat> images;
  for (const std::string &path : imagesPaths)
  {
    if (printing)
    {
      std::cout << "LoadImages: Loading image from path: " << path << std::endl;
    }
    cv::Mat img = cv::imread(path);
    std::vector<int> valueRange = {0, 255};
    double minVal;
    double maxVal;
    cv::minMaxLoc(img, &minVal, &maxVal);
    if (maxVal <= 1.0)
    {
      valueRange[0] = round(255.0 * (double)minVal);
      valueRange[1] = round(255.0 * (double)maxVal);
      if (printing)
      {
        std::cout << "LoadImages: image of type float, scaling to [" << valueRange[0] << ", " << valueRange[1] << "]" << std::endl;
      }
      cv::normalize(img, img, valueRange[1], valueRange[0], cv::NORM_MINMAX, CV_8U);
    }
    if ((matchFirstDimSize > -1) && (cv::countNonZero(matchFirstDimSize) > 0))
    {
      bool mismatchedDimensions = false;
      if (matchFirstDimSize != img.size[0])
      {
        mismatchedDimensions = true;
        // break;
      }
      if (mismatchedDimensions)
      {
        cv::rotate(img, img, cv::ROTATE_90_COUNTERCLOCKWISE);
      }
    }
    images.push_back(img);
  }
  return std::make_tuple(images, imagesPaths);
}

template<typename T>
void SaveToCSV(std::vector<T> data, std::string path="./data.csv") {
  using namespace std;

  fstream file;
  file.open(path, ios::out);

  for (int i = 0; i < data.size(); i++) {
    file << data[i] << "\n";
  }
  // file << "\n";
  file.close();
}

void SaveImage(const cv::Mat &image, const std::string &name="img", const std::string &fileType=".jpg", bool overwrite=false, bool addTimestamp=true, bool RGB2BGR=false) {
  using namespace std;
  using namespace cv;

  string filePath = "";
  if(addTimestamp) {
    time_t timestamp = time(NULL);
    struct tm datetime = *localtime(&timestamp);
    char dateFormatted[16];
    strftime(dateFormatted, 16, "%Y%m%d_%H%M%S", &datetime);
    filePath = name+"_"+dateFormatted+"_"+fileType;
  }
  else {
    filePath = name + fileType;
  }
  
  Mat _image = image;
  if(RGB2BGR) {
    cvtColor(_image, _image, COLOR_RGB2BGR);
  }

  //dont overwrite existing file
  if(!overwrite) {
    if(!filesystem::exists(filePath)) {
      imwrite(filePath, _image);
    }
  }
  else {
    imwrite(filePath, _image);
  }
}

cv::Mat Resize(const cv::Mat &inputImage, int newWidth, int newHeight, int interpolation = cv::INTER_LINEAR)
{
  cv::Mat image = inputImage.clone();
  if((inputImage.rows != newHeight) || (inputImage.cols != newWidth)) {
    cv::resize(image, image, cv::Size(newWidth, newHeight), 0, 0, interpolation);
  }
  return image;
}

cv::Mat ReduceResolution(cv::Mat inputImage, int maxRes = 1000, int interpolation = cv::INTER_LINEAR)
{
  cv::Mat image;
  if ((std::max(inputImage.rows, inputImage.cols) > maxRes) && (inputImage.rows > inputImage.cols))
  {
    int newHeight = maxRes;
    int newWidth = int(maxRes * (static_cast<double>(inputImage.cols) / inputImage.rows));
    cv::resize(inputImage.clone(), image, cv::Size(newWidth, newHeight), 0, 0, interpolation);
  }
  else if ((std::max(inputImage.rows, inputImage.cols) > maxRes) && (inputImage.rows <= inputImage.cols))
  {
    int newHeight = int(maxRes * (static_cast<double>(inputImage.rows) / inputImage.cols));
    int newWidth = maxRes;
    cv::resize(inputImage.clone(), image, cv::Size(newWidth, newHeight), 0, 0, interpolation);
  }
  else
  {
    image = inputImage.clone();
  }
  return image;
}

std::tuple<cv::Mat, std::vector<cv::Point>> FindNonZero(cv::Mat A)
{
  using namespace std;
  using namespace cv;

  vector<Point> nz;
  findNonZero(A, nz);

  Mat nonzeros;
  for (Point p : nz)
  {
    nonzeros.push_back(A.at<float>(p));
  }
  return make_tuple(nonzeros, nz);
}

int index2Dto1D(int ind1, int ind2, int size1)
{
  return ind1 * size1 + ind2;
}

/// @brief Creates ROI from 4 corner points
cv::Rect CornersToRect(const cv::Point& topLeftCorner,
                         const cv::Point& topRightCorner,
                         const cv::Point& bottomLeftCorner,
                         const cv::Point& bottomRightCorner) {
  
  int topCut = std::min(topLeftCorner.x, topRightCorner.x);
  int bottomCut = std::max(bottomLeftCorner.x, bottomRightCorner.x);
  int leftCut = std::min(topLeftCorner.y, bottomLeftCorner.y);
  int rightCut = std::max(topRightCorner.y, bottomRightCorner.y);
  int width = rightCut-leftCut;
  int height = bottomCut-topCut;

  cv::Rect roi(leftCut,topCut, width,height);
  return roi;
}

cv::Mat CutImgToCorners(const cv::Mat &inputImage, std::vector<int> topLeftCorner, std::vector<int> topRightCorner, std::vector<int> bottomLeftCorner, std::vector<int> bottomRightCorner)
{
  int topCut = std::min(topLeftCorner[0], topRightCorner[0]);
  int bottomCut = std::max(bottomLeftCorner[0], bottomRightCorner[0]);
  int leftCut = std::min(topLeftCorner[1], bottomLeftCorner[1]);
  int rightCut = std::max(topRightCorner[1], bottomRightCorner[1]);
  int width = rightCut-leftCut;
  int height = bottomCut-topCut;

  cv::Rect roi(leftCut,topCut, width,height);
  cv::Mat cropped = cv::Mat(inputImage.clone(), roi);
  return cropped;
}

cv::Mat CutImgToCorners(const cv::Mat &inputImage,
                         const cv::Point& topLeftCorner,
                         const cv::Point& topRightCorner,
                         const cv::Point& bottomLeftCorner,
                         const cv::Point& bottomRightCorner)
{
  int topCut = std::min(topLeftCorner.x, topRightCorner.x);
  int bottomCut = std::max(bottomLeftCorner.x, bottomRightCorner.x);
  int leftCut = std::min(topLeftCorner.y, bottomLeftCorner.y);
  int rightCut = std::max(topRightCorner.y, bottomRightCorner.y);
  int width = rightCut-leftCut;
  int height = bottomCut-topCut;

  cv::Rect roi(leftCut,topCut, width,height);
  cv::Mat cropped = cv::Mat(inputImage.clone(), roi);
  return cropped;
}

cv::Mat MatchResolution(const cv::Mat& imageRef, const cv::Mat& image2, cv::InterpolationFlags interp=cv::InterpolationFlags::INTER_LINEAR) {
  cv::Mat result;
  if(imageRef.size() != image2.size()){
    result = Resize(image2, imageRef.size().width, imageRef.size().height, interp);
  }
  else {
    result = image2;
  }
  return result;
}

cv::Mat MatchResolution(const cv::Mat& image, const cv::Size2i& dimensions, cv::InterpolationFlags interp=cv::InterpolationFlags::INTER_LINEAR) {
  cv::Mat result;
  if((image.rows != dimensions.height) || (image.cols != dimensions.width)){
    result = Resize(image, dimensions.width, dimensions.height, interp);
  }
  else {
    result = image.clone();
  }
  return result;
}

/// @brief Function that prints figure in image format to screen
/// @param figure cv::Mat image to plot
/// @param title title of figure
/// @param pause time in miliseconds to pause program execution when plotting
void show(const cv::Mat& figure, const std::string &title="Figure", int pause=0, int width=-1, int height=-1) {
  using namespace std;
  using namespace cv;

  int figure_width, figure_height;

  if((width>0) && (height>0)) {
    figure_width = width;
    figure_height = height;
  }
  else {
    figure_width = (int)(0.8*(double)1920);
    figure_height = (int)(0.8*(double)1080);
  }

  Mat figure_ = Resize(figure, figure_width, figure_height, cv::INTER_LINEAR);
  imshow(title, figure_);
  waitKey(pause);
}

template<typename T>
T max(cv::Mat arr, T matType=(float)0.0f) {
  double maxVal, minVal;
  cv::minMaxLoc(arr, &minVal, &maxVal);
  double result = maxVal;
  return (T)result;
}

float max(std::vector<cv::Mat> vec) {
  double maxVal, minVal;
  cv::minMaxLoc(vec[0], &minVal, &maxVal);
  double result = maxVal;
  if(vec.size()>1) {
    for(int i=1;i<vec.size();i++){
      cv::minMaxLoc(vec[i], &minVal, &maxVal);
      result = std::max(result, maxVal);
    }
  }

  return (float)result;
}

template <class T>
T variance(std::valarray<T> const &varr) {
    T mean = varr.sum() / varr.size();
    std::valarray<T> distance = varr - mean;
    distance = distance * distance;
    return distance.sum() / distance.size();
}

cv::Mat add(const cv::Mat& array, const cv::Mat& array2) {
  cv::Mat result;
  if(array.channels() == array2.channels()) {
    cv::Mat delta;
    cv::add(array, array2, delta);
    result = delta;
  }
  else if(array.channels() > array2.channels()) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < array.channels(); i++) {
      cv::Mat delta;
      cv::Mat channel;
      extractChannel(array, channel, i);
      cv::add(channel, array2, delta);
      vec.push_back(delta);
    }
    result = to3dMat(vec);
  }
  else if(array.channels() < array2.channels()) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < array2.channels(); i++) {
      cv::Mat delta;
      cv::Mat channel;
      extractChannel(array2, channel, i);
      cv::add(array, channel, delta);
      vec.push_back(delta);
    }
    result = to3dMat(vec);
  }

  return result;
}

cv::Mat add(const cv::Mat& array, float value2) {
  cv::Mat result;
  if(array.channels() == 1) {
    cv::add(array, value2, result);
  }
  else if(array.channels() > 1) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < array.channels(); i++) {
      cv::Mat delta;
      cv::Mat channel;
      extractChannel(array, channel, i);
      cv::add(channel, value2, delta);
      vec.push_back(delta);
    }
    result = to3dMat(vec);
  }

  return result;
}

std::vector<cv::Mat> subtract(const std::vector<cv::Mat>& array, const cv::Mat& img) {
  std::vector<cv::Mat> result;
  for (int i = 0; i < array.size(); i++) {
    cv::Mat delta;
    cv::subtract(array[i], img, delta);
    result.push_back(delta);
  }
  return result;
}

cv::Mat subtract(const cv::Mat& array, const cv::Mat& array2) {
  cv::Mat result;
  if(array.channels() == array2.channels()) {
    cv::Mat delta;
    cv::subtract(array, array2, delta);
    result = delta;
  }
  else if(array.channels() > array2.channels()) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < array.channels(); i++) {
      cv::Mat delta;
      cv::Mat channel;
      extractChannel(array, channel, i);
      cv::subtract(channel, array2, delta);
      vec.push_back(delta);
    }
    result = to3dMat(vec);
  }
  else if(array.channels() < array2.channels()) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < array2.channels(); i++) {
      cv::Mat delta;
      cv::Mat channel;
      extractChannel(array2, channel, i);
      cv::subtract(array, channel, delta);
      vec.push_back(delta);
    }
    result = to3dMat(vec);
  }

  return result;
}

cv::Mat subtract(const cv::Mat& array, float value2) {
  cv::Mat result;
  if(array.channels() == 1) {
    cv::subtract(array, value2, result);
  }
  else if(array.channels() > 1) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < array.channels(); i++) {
      cv::Mat delta;
      cv::Mat channel;
      extractChannel(array, channel, i);
      cv::subtract(channel, value2, delta);
      vec.push_back(delta);
    }
    result = to3dMat(vec);
  }

  return result;
}

cv::Mat multiply(const cv::Mat& _array, const cv::Mat& _array2) {
  cv::Mat array,array2,result;
  _array.convertTo(array,  CV_32F);
  _array2.convertTo(array2, CV_32F);

  if(array.channels() == array2.channels()) {
    cv::Mat value;
    cv::multiply(array, array2, value);
    result = value;
  }
  else if(array.channels() > array2.channels()) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < array.channels(); i++) {
      cv::Mat value;
      cv::Mat channel;
      extractChannel(array, channel, i);
      cv::multiply(channel, array2, value);
      vec.push_back(value);
    }
    result = to3dMat(vec);
  }
  else if(array.channels() < array2.channels()) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < array2.channels(); i++) {
      cv::Mat value;
      cv::Mat channel;
      extractChannel(array2, channel, i);
      cv::multiply(array, channel, value);
      vec.push_back(value);
    }
    result = to3dMat(vec);
  }

  return result;
}

cv::Mat multiply(const cv::Mat& array, float value2) {
  using namespace std;
  using namespace cv;
      
  Mat array2;
  array.convertTo(array2, CV_32F);
  cv::Mat result;
  if(array.channels() == 1) {
    cv::Mat value = Mat::ones(array2.rows, array2.cols, CV_32F);
    cv::multiply(array, value, value, (double)value2);
    result = value;
  }
  else if(array.channels() > 1) {
    std::vector<cv::Mat> vec;
    cv::split(array2, vec);
    for (int i = 0; i < array2.channels(); i++) {
      cv::Mat value = Mat::ones(array2.rows, array2.cols, CV_32F);
      cv::Mat channel = vec[i];
      cv::multiply(channel, value, value, (double)value2);
      vec.push_back(value);
    }
    result = to3dMat(vec);
  }

  result.convertTo(result, array.type());
  return result;
}

cv::Mat divide(const cv::Mat& array, const cv::Mat& array2) {
  cv::Mat result;
  if(array.channels() == array2.channels()) {
    cv::Mat value;
    cv::divide(array, array2, value);
    result = value;
  }
  else if(array.channels() > array2.channels()) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < array.channels(); i++) {
      cv::Mat value;
      cv::Mat channel;
      extractChannel(array, channel, i);
      cv::divide(channel, array2, value);
      vec.push_back(value);
    }
    result = to3dMat(vec);
  }
  else if(array.channels() < array2.channels()) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < array2.channels(); i++) {
      cv::Mat value;
      cv::Mat channel;
      extractChannel(array2, channel, i);
      cv::divide(array, channel, value);
      vec.push_back(value);
    }
    result = to3dMat(vec);
  }

  return result;
}

cv::Mat divide(const cv::Mat& array, float value2) {
  using namespace std;
  using namespace cv;

  Mat array2;
  array.convertTo(array2, CV_32F);
  cv::Mat result;
  if(array.channels() == 1) {
    cv::Mat value = Mat::ones(array2.rows, array2.cols, CV_32F);
    cv::divide(array, value, value, 1.0/(double)(value2));
    result = value;
  }
  else if(array.channels() > 1) {
    std::vector<cv::Mat> channels;
    cv::split(array2, channels);
    for (int i = 0; i < array2.channels(); i++) {
      cv::Mat value = Mat::ones(array2.rows, array2.cols, CV_32F);
      cv::divide(channels[i], value, channels[i], 1.0/(double)(value2));
    }
    result = to3dMat(channels);
  }

  result.convertTo(result, array.type());
  return result;
}

cv::Mat divide(float value2, const cv::Mat& array) {
  using namespace std;
  using namespace cv;

  cv::Mat result;
  if(array.channels() == 1) {
    cv::Mat value = multiply(Mat::ones(array.size(), array.type()), value2);
    cv::divide(value, array, value);
    result = value;
  }
  else if(array.channels() > 1) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < array.channels(); i++) {
      cv::Mat channel;
      extractChannel(array, channel, i);
      cv::Mat value = multiply(Mat::ones(array.size(), array.type()), value2);
      cv::divide(value, channel, value);
      vec.push_back(value);
    }
    result = to3dMat(vec);
  }

  return result;
}

template <template <typename> class Container, 
          typename T1>
Container<T1> divide(const Container<T1>& array, float value2) {
  using namespace std;
  using namespace cv;

  Container<T1> result(array.size());
  for (int i = 0; i < array.size(); i++) {
    result[i] = array[i] / value2;
  }

  return result;
}

cv::Mat Sum(const std::vector<cv::Mat>& imgs2D) {
//Function takes vector of Mats and returns 2d sum(sum along all Mats)
  using namespace std;
  using namespace cv;
  cv::Mat img = cv::Mat::zeros(imgs2D[0].rows, imgs2D[0].cols, CV_32F);
  for(int c=0; c<imgs2D.size(); c++) {
    img = add(img,imgs2D[c]);
  }
  return img;
}

template<typename T>
cv::Mat patchNaNs(const cv::Mat& array, T fillValue, T minValue=(T)0.0, T maxValue=(T)1.0) {
  using namespace std;
  using namespace cv;

  std::vector<cv::Mat> channels;
  if(array.channels() > 1) {
    cv::split(array.clone(), channels);
  }
  else {
    channels.push_back(array.clone());
  }

  T negativeInf = -numeric_limits<T>::infinity();
  T positiveInf = numeric_limits<T>::infinity();

  for(int c=0; c<channels.size(); c++) {
    #pragma omp parallel for
    for(int i=0; i<array.rows; i++) {
      for(int j=0; j<array.cols; j++) {
        if(isnan(channels[c].at<T>(i,j))) {
          channels[c].at<T>(i,j) = fillValue;
        }
        else if(channels[c].at<T>(i,j) <= negativeInf) {
          channels[c].at<T>(i,j) = (T)minValue;
        }
        else if(channels[c].at<T>(i,j) >= positiveInf) {
          channels[c].at<T>(i,j) = (T)maxValue;
        }
      }
    }
  }
  return to3dMat(channels);
}

template <template <typename> class Container, 
          typename T1 >
Container<T1> NormalizeTo_0_1(const Container<T1>& array) {
    Container<T1> result(array.size());
    std::copy(std::begin(array), std::end(array), std::begin(result));

    T1 minVal = *std::min_element(std::begin(result), std::end(result));
    T1 maxVal = *std::max_element(std::begin(result), std::end(result));
    T1 delta2 = maxVal - minVal;

    if((minVal == (T1)0.0)) {
      if((maxVal == (T1)1.0) || (maxVal == minVal)) {
        return result;
      }
    }
    else if((maxVal == (T1)1.0)) {
      if((minVal == (T1)0.0) || (maxVal == minVal)) {
        return result;
      }
    }

    if(delta2 == (T1)0.0) {
      if(maxVal >= (T1)1.0) { delta2 = maxVal; }
      else if(maxVal == (T1)0.0) { delta2 = (T1)1.0; }
      else if(maxVal < (T1)0.0) { delta2 = -maxVal; }
    }

    for (int i = 0; i < result.size(); i++) {
        result[i] = (result[i] - minVal) / delta2; //(array[i] - minVal) / (maxVal - minVal);
    }
    return result;
}

std::vector<cv::Mat> NormalizeTo_0_1(const std::vector<cv::Mat>& array, int axis=2) {
    // Normalize to range [0.0; 1.0] across specified axis
    using namespace std;
    using namespace cv;

    std::vector<cv::Mat> result(array.size());
    cv::Mat min2D;
    cv::Mat max2D;
    //TODO: implement oder axis
    if(axis==2) {
      array[0].convertTo(min2D, CV_32F);
      array[0].convertTo(max2D, CV_32F);
      cv::Mat current2D; //cv::Mat::zeros(array[i].rows, array[i].cols, cv::CV_32F);
      for (int i = 1; i < array.size(); i++) {
        // current2D = array[i];
        array[i].convertTo(current2D, CV_32F);
        cv::min(current2D.clone(),min2D.clone(),min2D);
        cv::max(current2D.clone(),max2D.clone(),max2D);        
      }

      Mat stack3D = to3dMat(array);
      stack3D.convertTo(stack3D, CV_32F);
      Mat delta1 = subtract(stack3D, min2D);
      Mat delta2 = subtract(max2D, min2D); 
      stack3D = divide(delta1, delta2);
      result = toVecMat(stack3D);
    }

    return result;
}

cv::Mat NormalizeTo_0_1(const cv::Mat& array) {
    // Normalize to range [0.0; 1.0]
    using namespace std;
    using namespace cv;

    cv::Mat result;
    array.convertTo(result, CV_32F);
    double _minVal, _maxVal;
    cv::minMaxLoc(result, &_minVal, &_maxVal);
    float minVal = (float)_minVal;
    float maxVal = (float)_maxVal;

    if((minVal == (float)0.0f)) {
      if((maxVal == (float)1.0f) || (maxVal == minVal)) {
        return result;
      }
    }
    else if((maxVal == (float)1.0f)) {
      if((minVal == (float)0.0f) || (maxVal == minVal)) {
        return result;
      }
    }

    Mat delta1 = subtract(result, minVal);
    float delta2 = maxVal - minVal;
    if(delta2 == (float)0.0f) {
      if(maxVal >= (float)1.0f) { delta2 = maxVal; }
      else if(maxVal == (float)0.0f) { delta2 = (float)1.0f; }
      else if(maxVal < (float)0.0f) { delta2 = -maxVal; }
    }
    result = divide(delta1, delta2);
    // result[i] = (array[i] - minVal) / (maxVal - minVal);

    return result;
}

enum ImgReplaceMode {
  minimum,
  maximum
};

cv::Mat Replace(const cv::Mat& img, float newValue, ImgReplaceMode mode) {
  using namespace std;
  using namespace cv;

  Mat m = img;

  switch(mode) {
    case ImgReplaceMode::minimum:
      vector<int> newShape {img.rows*img.cols, img.channels()};
      vector<int> oryginalShape {img.rows, img.cols}; //oryginalShape {img.rows, img.cols, img.channels()};
      m = m.reshape(1, newShape);
      double minVal;
      double maxVal;
      int minIdx[3];
      int maxIdx[3];
      for (int i = 0; i<m.rows; i++)
      {      
        Mat point = m.row(i);        
        minMaxIdx(point, &minVal, &maxVal, minIdx, maxIdx);
        m.at<float>(i, minIdx[1]) = newValue;
      }
      m = m.reshape(img.channels(), oryginalShape);
      break;
  }

  return m;
}

template<typename T>
T LimitToRange(const T &x, const auto &lowBounds, const auto &upBounds)
{
  auto x_limited = x;
  if (x_limited < (T)lowBounds)
  {
    x_limited = (T)lowBounds;
  }
  else if (x_limited > (T)upBounds)
  {
    x_limited = (T)upBounds;
  }
  return x_limited;
}

template<typename T>
std::vector<T> LimitToRange(const std::vector<T> &x, const auto &lowBounds, const auto &upBounds)
{
  auto x_limited = x;
  for (int i = 0; i < x.size(); i++)
  {
    if (x_limited[i] < (T)lowBounds[i])
    {
      x_limited[i] = (T)lowBounds[i];
    }
    else if (x_limited[i] > (T)upBounds[i])
    {
      x_limited[i] = (T)upBounds[i];
    }
  }
  return x_limited;
}

/// @brief Replaces max values to min, min to max etc
/// @param mask Multichannel mask is treated as 1 big mask (not prosessing each channel separetly)
/// @return Mask of input size, float32 type and inverted values
cv::Mat invertMask(const cv::Mat &mask) {
  using namespace std;
  using namespace cv;

  Mat invMask = mask.clone();
  invMask.convertTo(invMask, CV_32F);
  double minVal,maxVal;
  minMaxLoc(invMask, &minVal, &maxVal);
  // invMask = subtract(invMask, (float)minVal);
  absdiff(invMask, (float)maxVal, invMask);

  return invMask;
}

/// @brief Compares and thresholds each channel of image to specified color. Default mask value is 1.0f, matching regions are set to 0.0f.
/// @param image input multichannel image
/// @param color optional: input multichannel color (default is 0)
/// @return Mask with size of image (1 channel), float32 type and values in range [0.0f; 1.0f] inclusive
cv::Mat maskFromColor(const cv::Mat &image, std::vector<int> color={0,0,0}) {
  using namespace std;
  using namespace cv;

  Mat imageGray, mask, current;
  vector<Mat> channels;
  mask = Mat::ones(image.rows, image.cols, CV_32F);

  cv::split(image.clone(), channels);
  for (int i = 0; i < channels.size(); i++)
  {
    double res = cv::threshold(channels[i], current, (double)color[i], 255.0, THRESH_BINARY);
    current.convertTo(current, CV_32F);
    mask = multiply(mask, current);
  }
  mask = NormalizeTo_0_1(mask);
  return mask;
}

/// @brief Divides each channel of image1 by image2 and checks if value is within detectionRatio. Default mask value is 1.0f, matching regions are set to 0.0f.
cv::Mat maskFromChange(const cv::Mat &image1, const cv::Mat &image2, float detectionRatio) {
  using namespace std;
  using namespace cv;

  Mat imageGray, mask, current;
  vector<Mat> channels;
  mask = Mat::ones(image1.rows, image1.cols, CV_32F);
  current = Mat::ones(image1.rows, image1.cols, CV_32F);
  Mat image1_, image2_;
  image1.convertTo(image1_, CV_32F);
  image2.convertTo(image2_, CV_32F);
  Mat result = divide(image1_, image2_);

  cv::split(result, channels);
  for (int i = 0; i < channels.size(); i++)
  {
    current = Mat::ones(image1.rows, image1.cols, CV_32F);
    
    if(detectionRatio < 1.0f)
    {
      #pragma omp parallel for
      for(int r = 0; r < result.rows; r++)
      {
        for(int c = 0; c < result.cols; c++) {
          if(channels[i].at<float>(r,c) <= detectionRatio) {
            current.at<float>(r,c) = 0.0f;
          }
        }        
      }      
    }
    if(detectionRatio > 1.0f)
    {
      #pragma omp parallel for
      for(int r = 0; r < result.rows; r++)
      {
        for(int c = 0; c < result.cols; c++) {
          if(channels[i].at<float>(r,c) >= detectionRatio) {
            current.at<float>(r,c) = 0.0f;
          }
        }        
      }      
    }

    mask = multiply(mask, current);
  }
  mask = NormalizeTo_0_1(mask);
  return mask;
}

/// @brief Divides each channel of image1 by image2 and checks if change is higher than allowedRatio. Matching pixels are limited to allowedRatio.
cv::Mat limitChange(const cv::Mat &image1, const cv::Mat &image2, float allowedRatio) {
  using namespace std;
  using namespace cv;

  Mat clampedImage, current;
  vector<Mat> channels, channelsImg1, channelsImg2, channelsClampedImage;
  clampedImage = Mat::ones(image1.rows, image1.cols, CV_32FC3);
  current = Mat::ones(image1.rows, image1.cols, CV_32F);
  Mat image1_, image2_;
  image1.clone().convertTo(image1_, CV_32F);
  image2.clone().convertTo(image2_, CV_32F);
  Mat result = divide(image1_, image2_);

  cv::split(result, channels);
  cv::split(image1_, channelsImg1);
  cv::split(image2_, channelsImg2);
  
  for (int i = 0; i < channels.size(); i++)
  {
    current = channelsImg2[i];
    
    if(allowedRatio < 1.0f)
    {
      #pragma omp parallel for
      for(int r = 0; r < result.rows; r++)
      {
        for(int c = 0; c < result.cols; c++) {
          if(channels[i].at<float>(r,c) <= allowedRatio) {
            current.at<float>(r,c) = channelsImg1[i].at<float>(r,c)*allowedRatio;
          }
          else if(channels[i].at<float>(r,c) > (1.0f/allowedRatio)) {
            current.at<float>(r,c) = channelsImg1[i].at<float>(r,c)/allowedRatio;
          }
        }        
      }      
    }
    if(allowedRatio > 1.0f)
    {
      #pragma omp parallel for
      for(int r = 0; r < result.rows; r++)
      {
        for(int c = 0; c < result.cols; c++) {
          if(channels[i].at<float>(r,c) >= allowedRatio) {
            current.at<float>(r,c) = channelsImg1[i].at<float>(r,c)/allowedRatio;
          }
          else if(channels[i].at<float>(r,c) < (1.0f/allowedRatio)) {
            current.at<float>(r,c) = channelsImg1[i].at<float>(r,c)*allowedRatio;
          }
        }        
      }      
    }

    channelsClampedImage.push_back(current);
  }

  cv::merge(channelsClampedImage, clampedImage);
  return clampedImage;
}

double CompareImg(const cv::Mat &imageRef, const cv::Mat &imageTest, double area = 1.0)
{

  using namespace std;
  using namespace cv;

  double area2 = area;
  if(area > 1.0) {
    area2 = area2 / 100.0;
  }
  if(area < 0.0) {
    area2 = 1.0;
  }

  Mat imageTest2 = MatchResolution(imageRef,imageTest);
  // int xCut = (int)((1.0 - area2) * 0.5 * imageRef.rows);
  // int yCut = (int)((1.0 - area2) * 0.5 * imageRef.cols);
  int xCut = (int)((1.0 - area2) * 0.5 * imageRef.cols);
  int yCut = (int)((1.0 - area2) * 0.5 * imageRef.rows);
  std::vector<int> topLeftCorner = {0 + yCut, 0 + xCut};
  std::vector<int> topRightCorner = {0 + yCut, imageRef.cols - xCut};
  std::vector<int> bottomLeftCorner = {imageRef.rows - yCut, 0 + xCut};
  std::vector<int> bottomRightCorner = {imageRef.rows - yCut, imageRef.cols - xCut};
  cv::Mat image1 = CutImgToCorners(imageRef, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
  cv::Mat image2 = CutImgToCorners(imageTest2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
  if (image1.channels() > 1)
  {
    cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
  }
  if (image2.channels() > 1)
  {
    cv::cvtColor(image2, image2, cv::COLOR_BGR2GRAY);
  }

  Mat errorMSE;
  absdiff(image1, image2, errorMSE);
  errorMSE.convertTo(errorMSE, CV_32F);
  errorMSE = errorMSE.mul(errorMSE);
  sqrt(errorMSE, errorMSE); // sqrt(errorMSE / (image1.rows * image1.cols), errorMSE);
  Scalar s = sum(errorMSE); // sum per channel
  double mse = (s.val[0] + s.val[1] + s.val[2]) / (double)(image1.channels() * image1.rows * image1.cols);
  return mse;
}

std::tuple<double, double, cv::Mat> CompareImg2(const cv::Mat &refImg, const cv::Mat &distImg, const cv::Mat &undistorted,
                                                const cv::Mat &mtx, const cv::Mat &dist, const cv::Size &size, bool alignCenter,
                                                double checkArea = 1.0)
{
  cv::Mat refImgGray;
  if (refImg.channels() > 1)
  {
    cv::cvtColor(refImg, refImgGray, cv::COLOR_BGR2GRAY);
  }
  else
  {
    refImgGray = distImg;
  }
  cv::Mat undistortedImgGray;
  if (undistorted.channels() > 1)
  {
    cv::cvtColor(undistorted, undistortedImgGray, cv::COLOR_BGR2GRAY);
  }
  else
  {
    undistortedImgGray = distImg;
  }
  cv::Mat distoredImgGray;
  if (distImg.channels() > 1)
  {
    cv::cvtColor(distImg, distoredImgGray, cv::COLOR_BGR2GRAY);
  }
  else
  {
    distoredImgGray = distImg;
  }
  distoredImgGray = distoredImgGray + 1;

  auto undistoredGray = Undistort(distoredImgGray, mtx, dist,
                                  size.width, size.height,
                                  cv::INTER_LINEAR, 1.0, alignCenter);

  cv::Mat mask; // = undistoredGray>0; unknown pixels have value 0, and known >0
  double res = cv::threshold(undistoredGray, mask, 1.0, 255.0, cv::THRESH_BINARY);
  cv::Mat oryginalMasked;
  cv::Mat udistortedMasked;
  cv::bitwise_and(refImgGray, mask, oryginalMasked);
  cv::bitwise_and(undistortedImgGray, mask, udistortedMasked);
  auto mse1 = CompareImg(oryginalMasked, udistortedMasked, checkArea); // mse of known pixels
  auto mse2 = CompareImg(refImgGray, undistortedImgGray, checkArea);   // mse of all pixels
  return std::make_tuple(mse1, mse2, mask);
}

/**
 * @brief Compare difference of image2 to reference image1.
 * 
 * @param image1_ reference image
 * @param image2_ target image
 * @param printing optionally print results to console
 * @return std::tuple<double, double> {mse, psnr}
 */
std::tuple<double, double> CompareMetrics(const cv::Mat& image1_, const cv::Mat& image2_, bool printing = false)
{
  using namespace cv;
  Mat image1; // = image1_.clone();
  Mat image2; // = image2_.clone();
  image1_.convertTo(image1,CV_32F);
  image2_.convertTo(image2,CV_32F);
  image2 = MatchResolution(image1, image2);

  if ((image1.channels() > 2) || (image2.channels() > 2))
  {
    cv::Mat redDiff, greenDiff, blueDiff;
    cv::subtract(image1, image2, redDiff, noArray(), CV_32F);
    cv::subtract(image1, image2, greenDiff, noArray(), CV_32F);
    cv::subtract(image1, image2, blueDiff, noArray(), CV_32F);
    cv::pow(redDiff, 2, redDiff);
    cv::pow(greenDiff, 2, greenDiff);
    cv::pow(blueDiff, 2, blueDiff);
    auto sumR = cv::sum(redDiff);
    auto sumG = cv::sum(greenDiff);
    auto sumB = cv::sum(blueDiff);
    double redE = cv::sqrt(sumR[0] / (image1.rows * image1.cols));
    double greenE = cv::sqrt(sumG[0] / (image1.rows * image1.cols));
    double blueE = cv::sqrt(sumB[0] / (image1.rows * image1.cols));
    double errorMSE = (redE + greenE + blueE) / 3.0;
    Mat image1Gray=image1;
    Mat image2Gray=image2;
    if(image1.channels()>1){
      cv::cvtColor(image1, image1Gray, cv::COLOR_BGR2GRAY);
    }
    if(image2.channels()>1){
      cv::cvtColor(image2, image2Gray, cv::COLOR_BGR2GRAY);
    }
    // double psnr = cv::PSNR(image1, image2);
    double psnr = cv::PSNR(image1Gray, image2Gray);
    if (printing)
    {
      std::cout << "mse equals " << errorMSE << std::endl;
      std::cout << "PSNR equals " << psnr << std::endl;
    }
    return std::make_tuple(errorMSE, psnr);
  }
  else
  {
    cv::Mat diff;
    cv::subtract(image1, image2, diff, noArray(), CV_32F);
    cv::pow(diff, 2, diff);
    auto sum = cv::sum(diff);
    double errorMSE = cv::sqrt(sum[0] / (image1.rows * image1.cols));
    double psnr = cv::PSNR(image1, image2);
    if (printing)
    {
      std::cout << "mse equals " << errorMSE << std::endl;
      std::cout << "PSNR equals " << psnr << std::endl;
    }
    return std::make_tuple(errorMSE, psnr);
  }
}

/// @brief Point detector (matching between images) using specified method
/// @param printing show more information, used for debugging
/// @return pairs of points in img1 and img2
std::pair<cv::Mat, cv::Mat> DetectFeatures(const cv::Mat &img1, const cv::Mat &img2, int nfeatures=32000, 
                                      const cv::Mat &mask=cv::Mat(), const Detector &method=Detector::orb,
                                      float scaleFactor=1.5f,int nlevels=5,float Lowes=0.77f,
                                      int K=2, bool printing=false) {
  //Detect points on 2 images and return good matches
  using namespace std;
  using namespace cv;

  Ptr<Feature2D> detector;
  if(method==Detector::orb) {
    detector = ORB::create(nfeatures, scaleFactor, nlevels);
  }
  else if(method==Detector::sift) {
    detector = SIFT::create(nfeatures, 3, 0.02, 15.0, 1.2);
  }
  else if(method==Detector::kaze) {
    detector = KAZE::create();
  }
  else if(method==Detector::akaze) {
    detector = AKAZE::create();
  }
  else {
    detector = ORB::create(nfeatures);
  }

  Mat grayImg1;
  if(img1.channels() > 1) {
    cvtColor(img1.clone(), grayImg1, COLOR_BGR2GRAY);
  }
  else {
    grayImg1 = img1.clone();
  }
  // blur(grayImg1,grayImg1, cv::Size(3,3));

  vector<KeyPoint> keypoints1;
  Mat descriptors1;
  if(mask.rows > 0) {
    detector->detectAndCompute(grayImg1, mask, keypoints1, descriptors1);
  }
  else { 
    detector->detectAndCompute(grayImg1, noArray(), keypoints1, descriptors1); 
  }

  Mat grayImg2;
  if(img2.channels() > 1) {
    cvtColor(img2, grayImg2, COLOR_BGR2GRAY);
  }
  else {
    grayImg2 = img2;
  }
  // blur(grayImg2,grayImg2, cv::Size(3,3));


  vector<KeyPoint> keypoints2;
  Mat descriptors2;
  if(mask.rows > 0) {
    detector->detectAndCompute(grayImg2, mask, keypoints2, descriptors2); 
  }
  else { 
    detector->detectAndCompute(grayImg2, noArray(), keypoints2, descriptors2); 
  }

  if(descriptors1.empty() || descriptors1.rows < 8) { 
    if(VERBOSITY > 0 ) {
      cout<<"DetectFeatures: error - empty descriptor of img1"<<endl;
    }
    return make_pair(cv::Mat{}, cv::Mat{});
  }
  if(descriptors2.empty() || descriptors1.rows < 8) { 
    cout<<"DetectFeatures: error - empty descriptor of img2"<<endl;
    return make_pair(cv::Mat{}, cv::Mat{});
  }
  descriptors1.convertTo(descriptors1, CV_32F);
  descriptors2.convertTo(descriptors2, CV_32F);

  //match points
  vector<vector<DMatch>> possibleMAtches;
  Ptr<FlannBasedMatcher> flann = FlannBasedMatcher::create();

  try
  {
    flann->knnMatch(descriptors1, descriptors2, possibleMAtches, K);
  }
  catch(const std::exception& e)
  {
    std::cerr << e.what() << '\n';
  }
  
  if(possibleMAtches.size() < 4) {
    if(VERBOSITY > 0) {
      cout << "DetectFeatures: WARNING: Failed to detect points" << endl;
    }
    return make_pair(cv::Mat{}, cv::Mat{});
  }

  vector<DMatch> matches;
  for (const auto &pair : possibleMAtches) {
      if (pair[0].distance < Lowes * pair[1].distance)
          matches.push_back(pair[0]);
  }

  Mat matchpoints1 = Mat::zeros(matches.size(), 2, CV_32F);
  Mat matchpoints2 = Mat::zeros(matches.size(), 2, CV_32F);

  for (int i = 0; i < matches.size(); ++i) {
      auto &match = matches[i];

      matchpoints1.at<float>(i, 0) = keypoints1[match.queryIdx].pt.x;
      matchpoints1.at<float>(i, 1) = keypoints1[match.queryIdx].pt.y;

      matchpoints2.at<float>(i, 0) = keypoints2[match.trainIdx].pt.x;
      matchpoints2.at<float>(i, 1) = keypoints2[match.trainIdx].pt.y;
  }

  //make sure number of points is equal
  int n = min(matchpoints1.rows, matchpoints2.rows);
  Mat matchpoints1_ = Mat(n, matchpoints1.cols, CV_32F);
  Mat matchpoints2_ = Mat(n, matchpoints2.cols, CV_32F);
  if(n < 4) {
    if(VERBOSITY > 0) {
      cout << "DetectFeatures:WARNING: Failed to detect points" << endl;
    }
    return make_pair(matchpoints1_, matchpoints2_);
  }
  matchpoints1.col(0).copyTo(matchpoints1_.col(0));
  matchpoints1.col(1).copyTo(matchpoints1_.col(1));
  matchpoints2.col(0).copyTo(matchpoints2_.col(0));
  matchpoints2.col(1).copyTo(matchpoints2_.col(1));

  //-- Draw matches
  if(printing==true) {
    Mat img_matches;
    drawMatches( img1.clone(), keypoints1, img2, keypoints2, matches, img_matches, Scalar::all(-1),
                Scalar::all(-1), std::vector<char>(), DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS );
    imshow("Good Matches", img_matches );
    waitKey(1000);
  }

  return make_pair(matchpoints1_, matchpoints2_);
}

/// @brief Alternative point detector (matching between images) based on corners
/// @param printing show more information, used for debugging
/// @return pairs of points in img1 and img2
std::pair<cv::Mat, cv::Mat> DetectFeatures2(const cv::Mat &img1, const cv::Mat &img2,
                                      int nfeatures=30,double qualityLevel=0.01,double minDistance=10,bool printing=false) {
  //Detect points on 2 images and return good matches
  using namespace std;
  using namespace cv;

  Mat grayImg1;
  if(img1.channels() > 1) {
    cvtColor(img1.clone(), grayImg1, COLOR_BGR2GRAY);
  }
  else {
    grayImg1 = img1.clone();
  }
  vector<Point> keypoints1;

  cv::goodFeaturesToTrack(grayImg1,keypoints1,nfeatures,qualityLevel,minDistance);

  Mat grayImg2;
  if(img2.channels() > 1) {
    cvtColor(img2, grayImg2, COLOR_BGR2GRAY);
  }
  else {
    grayImg2 = img2;
  }
  vector<Point> keypoints2;

  cv::goodFeaturesToTrack(grayImg2,keypoints2,nfeatures,qualityLevel,minDistance);

  Mat matchpoints1 = Mat::zeros(keypoints1.size(), 2, CV_32F);
  Mat matchpoints2 = Mat::zeros(keypoints2.size(), 2, CV_32F);
  //make sure number of points is equal
  int n = min(keypoints1.size(), keypoints2.size());

  for (int i = 0; i < n; ++i) {
      matchpoints1.at<float>(i, 0) = (float)keypoints1[i].x;
      matchpoints1.at<float>(i, 1) = (float)keypoints1[i].y;
      matchpoints2.at<float>(i, 0) = (float)keypoints2[i].x;
      matchpoints2.at<float>(i, 1) = (float)keypoints2[i].y;
  }

  if(n < 4) {
    if(VERBOSITY > 0) {
      cout << "DetectFeatures:WARNING: Failed to detect points" << endl;
    }
    return make_pair(matchpoints1, matchpoints2);
  }

  //-- Draw matches
  if(printing==true) {
    Mat img_matches = img1.clone();
    for (int i = 0; i < keypoints1.size(); i++)
    {
      drawMarker(img_matches, keypoints1[i],Scalar::all(-1));
      drawMarker(img_matches, keypoints2[i],Scalar::all(255));
    }    
    imshow("Good Matches", img_matches );
    waitKey(1000);
  }

  return make_pair(matchpoints1, matchpoints2);
}

cv::Mat AlignImageToImage(const cv::Mat &orygImage, const cv::Mat &inputImage, const cv::Mat &M, 
                          cv::Mat* outMask=nullptr,int borderMode=cv::BORDER_CONSTANT,
                          int flags=cv::INTER_CUBIC, bool affine=false) {
  using namespace std;
  using namespace cv;
  

  Mat H = M.clone();

  if (affine) {
      H = H.rowRange(0, 2);
  }

  int h = orygImage.rows, w = orygImage.cols, c = orygImage.channels();
  cv::Mat unwarped;
  if (borderMode == cv::BORDER_CONSTANT) {
      if (affine) {
          cv::warpAffine(inputImage, unwarped, H, cv::Size(w, h), flags, cv::BORDER_CONSTANT, 0);
      } else {
          cv::warpPerspective(inputImage, unwarped, H, cv::Size(w, h), flags, cv::BORDER_CONSTANT, 0);
      }
  } else {
      if (affine) {
          cv::warpAffine(inputImage, unwarped, H, cv::Size(w, h), flags, cv::BORDER_CONSTANT, 0);
      } else {
          cv::warpPerspective(inputImage, unwarped, H, cv::Size(w, h), flags, cv::BORDER_CONSTANT, 0);
      }
  }

  if (outMask != nullptr) {
      cv::Mat mask = (*outMask).clone();
      if (max(mask,(uchar)0) == (uchar)1) {
          mask *= (uchar)255;
      }
      if (affine) {
          cv::warpAffine(mask, mask, H, cv::Size(w, h), flags, cv::BORDER_CONSTANT, 0);
      } else {
          cv::warpPerspective(mask, mask, H, cv::Size(w, h), flags, cv::BORDER_CONSTANT, 0);
      }
      mask = mask > 0;
      (*outMask) = mask;
  }

  //check for improvement
  double mse1 = CompareImg(orygImage, inputImage, 0.5);
  double mse2 = CompareImg(orygImage, unwarped, 0.5);
  if(mse2 > mse1) {
    unwarped = inputImage.clone();
    if(VERBOSITY > 0) {
      cout << "AlignImageToImage: Warning: Failed to align image. Returning unchanged input image." <<endl;
    }
  }

  return unwarped;
}

cv::Mat AlignImageToImage(const cv::Mat &orygImage, const cv::Mat &inputImage,
                          int nfeatures=4000,float scaleFactor=1.5f,int nlevels=5,float Lowes=0.75f,int K=3, 
                          float ransacReprojThreshold=10.0f,cv::Mat* outMask=nullptr,int borderMode=cv::BORDER_CONSTANT,
                          int eccIter=0,bool affine=false, 
                          cv::Mat* M=nullptr) {
  using namespace std;
  using namespace cv;
  
  auto [pts0, pts1] = DetectFeatures(inputImage, orygImage, nfeatures,cv::Mat(),Detector::orb,scaleFactor,nlevels,Lowes,K);
  // auto [pts0, pts1] = DetectFeatures2(inputImage, orygImage,100,0.005,10.0,true);
  if((pts0.rows < 8) || (pts1.rows < 8)) {
    if(VERBOSITY > 0) {
      cout << "AlignImageToImage: WARNING: Failed to align images. Returning input." <<endl;
    }
    return inputImage.clone();
  }

  auto H = cv::findHomography(pts0, pts1, cv::RANSAC, ransacReprojThreshold);
  // auto H = cv::findHomography(pts0, pts1, cv::RHO, ransacReprojThreshold);

  int flags = cv::INTER_LANCZOS4;
  if (eccIter > 1) {
    Mat orygGray;
    Mat inputGray;
    if(orygImage.channels() > 1) {
      cv::cvtColor(orygImage.clone(), orygGray, cv::COLOR_BGR2GRAY);
    }
    else {
      orygGray = orygImage.clone();
    }
    if(inputImage.channels() > 1) {
      cv::cvtColor(inputImage.clone(), inputGray, cv::COLOR_BGR2GRAY);
    }
    else {
      inputGray = inputImage.clone();
    }
    flags = cv::INTER_LANCZOS4;
    H = cv::Mat(H).inv();
    H.convertTo(H, CV_32F);
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, eccIter, 1e-4);
    try
    {
      if (outMask != nullptr) {
        cv::findTransformECC(orygGray, inputGray, H, cv::MOTION_HOMOGRAPHY, criteria, *outMask);
      } else {
        cv::findTransformECC(orygGray, inputGray, H, cv::MOTION_HOMOGRAPHY, criteria);
      }
    }
    catch(const std::exception& e)
    {
      std::cerr << e.what() << '\n';
    }
    H = cv::Mat(H).inv();
    H.convertTo(H, CV_32F);
  } else {
    flags = cv::INTER_LANCZOS4;
  }

  Mat unwarped = AlignImageToImage(orygImage, inputImage, H,
                          outMask, borderMode, flags, affine);

  if(M != nullptr) {
    *M = H;
  }

  return unwarped;
}

cv::Mat AlignImageToImageRegions(const cv::Mat &orygImage, const cv::Mat &inputImage, cv::Size2i num_parts=cv::Size2i(2,2), 
                          cv::Mat* outMask=nullptr, int interpolation=cv::INTER_CUBIC)
{
  using namespace std;
  using namespace cv;
  
  int borderMode=cv::BORDER_CONSTANT;
  auto imageSize = orygImage.size();
  int width = (int)(imageSize.width / num_parts.width);
  int height = (int)(imageSize.height / num_parts.height);
  vector<vector<Rect>> rois;
  vector<vector<Mat>> images;
  vector<vector<Mat>> images2;
  vector<vector<Mat>> imagesAligned;
  rois.resize(num_parts.width);
  images.resize(num_parts.width);
  images2.resize(num_parts.width);
  imagesAligned.resize(num_parts.width);

  for (int i = 0; i < num_parts.width; i++)
  {
    for (int j = 0; j < num_parts.height; j++)
    {
      cv::Rect roi = cv::Rect((i*width),(j*height), width,height);
      // Mat truncated = orygImage(roi);
      // Mat truncated2 = inputImage(roi);
      Mat truncated;
      Mat truncated2;
      orygImage(roi).clone().copyTo(truncated);
      inputImage(roi).clone().copyTo(truncated2);
      rois[i].push_back(roi);
      images[i].push_back(truncated);
      images2[i].push_back(truncated2);
    }
  }
  
  for (int i = 0; i < num_parts.width; i++)
  {
    for (int j = 0; j < num_parts.height; j++)
    {
      Mat aligned;
      try
      {
        aligned = AlignImageToImage(images[i][j], images2[i][j], 
                                    1000,1.1f,16,0.7, 
                                    2, 5.0f );
      }
      catch(const std::exception& e)
      {
        try
        {
          aligned = AlignImageToImage(images[i][j], images2[i][j]);
        }
        catch(const std::exception& e)
        {
          aligned = images2[i][j];
          std::cerr << e.what() << '\n';
        }
      }
      imagesAligned[i].push_back(aligned);
    }
  }
  
  Mat result = Mat::zeros(imageSize, inputImage.type());
  for (int i = 0; i < num_parts.width; i++)
  {
    for (int j = 0; j < num_parts.height; j++)
    {
      // result(rois[i][j]) = imagesAligned[i][j];
      imagesAligned[i][j].copyTo(result(rois[i][j]));
    }
  }

  return result;
}

bool compare_Point2f_dim0(const cv::Point2f &a, const cv::Point2f &b) {
    return a.x > b.x;
}

bool compare_Point2f_dim1(const cv::Point2f &a, const cv::Point2f &b) {
    return a.y > b.y;
}

bool compare_pointVecf_dim0(const std::vector<float> &a, const std::vector<float> &b) {
    return a[0] > b[0];
}

bool compare_pointVecf_dim1(const std::vector<float> &a, const std::vector<float> &b) {
    return a[1] > b[1];
}

/// @brief Converts vector of points to K centers.
/// @param K number of clusters.
/// @param objectPoints2D_ vector of n-dimensional points.
/// @return K averaged centers of clusters.
cv::Mat simpleClusters(int K, std::vector<std::vector<float>> &objectPoints2D_)
{
  using namespace std;
  using namespace cv;

  valarray<valarray<float>> objectPoints2D = toValarray(objectPoints2D_);
  int cols = objectPoints2D[0].size();
  Mat centers = Mat::zeros(K, cols, CV_32F);
  int rows = objectPoints2D.size();
  int n = (int)((float)rows / (float)K);

  if(rows <= K) {
    for (int part = 0; part < rows; part++) {
      for (int i = 0; i < cols; i++)
      {
        centers.at<float>(part, i) = objectPoints2D[part][i];
      }
    }
    return centers;
  }

  for (int part = 0; part < K; part++)
  {
    cv::Mat mean_;
    valarray<float> zero((float)0.0f, cols); //std::vector<float> zero(0.0f, 0.0f);
    valarray<float> sum((float)0.0f, cols);
    for (int i = 0; i < n; i++)
    {
      sum = sum + objectPoints2D[part*n+i];
    }    

    for (int i = 0; i < cols; i++)
    {
      centers.at<float>(part, i) = sum[i] / n; //mean_point[i];
    }    
  }
  return centers;
}

//Split points into 4 parts {from top-left to bottom-right}. Average each part into K cluster centers.
cv::Mat cluster4way(int K, const std::vector<std::vector<float>> &objectPoints2D_, cv::Size2f imageSize)
{
  using namespace std;
  using namespace cv;

  if(objectPoints2D_.size() <= (K*4+2)) {
    return toMat(objectPoints2D_);
  }

  auto objectPoints2D = objectPoints2D_;
  std::sort(objectPoints2D.begin(), objectPoints2D.end(), compare_pointVecf_dim0);
  std::vector<std::vector<float>> points_top(&objectPoints2D[0], &objectPoints2D[objectPoints2D.size()/2]);
  std::vector<std::vector<float>> points_bot(&objectPoints2D[objectPoints2D.size()/2], &objectPoints2D[objectPoints2D.size()-1]);
  if(points_top.size() > K) {
    for (int i = K; i < points_top.size(); i++)
    {
      if(points_top[i][0] > (imageSize.height/2)) {
        vector<std::vector<float>> validPart(&points_top[0], &points_top[i]);
        points_top = validPart;
        break;
      }
    }
  }
  if(points_bot.size() > K) {
    for (int i = points_bot.size()-K-1; i >= 0; i--)
    {
      if(points_bot[i][0] < (imageSize.height/2)) {
        vector<std::vector<float>> validPart(&points_bot[i], &points_bot[points_bot.size()-1]);
        points_bot = validPart;
        break;
      }
    }
  }
  std::sort(points_top.begin(), points_top.end(), compare_pointVecf_dim1);
  std::sort(points_bot.begin(), points_bot.end(), compare_pointVecf_dim1);
  std::vector<std::vector<float>> points_top_left(&points_top[0], &points_top[points_top.size()/2]);  //top-left points
  std::vector<std::vector<float>> points_top_right(&points_top[points_top.size()/2], &points_top[points_top.size()-1]);  //top-right points
  std::vector<std::vector<float>> points_bot_left(&points_bot[0], &points_bot[points_bot.size()/2]);  //bottom-left points
  std::vector<std::vector<float>> points_bot_right(&points_bot[points_bot.size()/2], &points_bot[points_bot.size()-1]);  //bottom-right points
  if(points_top_left.size() > K) {
    for (int i = K; i < points_top_left.size(); i++)
    {
      if(points_top_left[i][1] > (imageSize.width/2)) {
        vector<std::vector<float>> validPart(&points_top_left[0], &points_top_left[i]);
        points_top_left = validPart;
        break;
      }
    }
  }
  if(points_top_right.size() > K) {
    for (int i = points_top_right.size()-K-1; i >= 0; i--)
    {
      if(points_top_right[i][1] < (imageSize.width/2)) {
        vector<std::vector<float>> validPart(&points_top_right[i], &points_top_right[points_top_right.size()-1]);
        points_top_right = validPart;
        break;
      }
    }
  }
  if(points_bot_left.size() > K) {
    for (int i = K; i < points_bot_left.size(); i++)
    {
      if(points_bot_left[i][1] > (imageSize.width/2)) {
        vector<std::vector<float>> validPart(&points_bot_left[0], &points_bot_left[i]);
        points_bot_left = validPart;
        break;
      }
    }
  }
  if(points_bot_right.size() > K) {
    for (int i = points_bot_right.size()-K-1; i >= 0; i--)
    {
      if(points_bot_right[i][1] < (imageSize.width/2)) {
        vector<std::vector<float>> validPart(&points_bot_right[i], &points_bot_right[points_bot_right.size()-1]);
        points_bot_right = validPart;
        break;
      }
    }
  }

  Mat centers_TL = simpleClusters(K, points_top_left); //top-left clusters
  Mat centers_TR = simpleClusters(K, points_top_right); //top-right clusters
  Mat centers_BL = simpleClusters(K, points_bot_left); //bottom-left clusters
  Mat centers_BR = simpleClusters(K, points_bot_right); //bottom-right clusters

  Mat clusters;
  vconcat(centers_TL, centers_TR, clusters);
  vconcat(clusters, centers_BL, clusters);
  vconcat(clusters, centers_BR, clusters);
  return clusters;
}

//Selects randomly K points.
cv::Mat getKRandPoints(int K, const std::vector<std::vector<float>> &objectPoints2D_)
{
  using namespace std;
  using namespace cv;

  if(objectPoints2D_.size() < K) {
    return toMat(objectPoints2D_);
  }
  if(K <= 0) {
    return toMat(objectPoints2D_);
  }

  int w = objectPoints2D_[0].size();
  int h = objectPoints2D_.size();
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution((float)0.0f,(float)1.0f);
  Mat clusters(0, w, cv::DataType<float>::type);
  for(int i = 0; i < K; i++)
  {
    if(distribution(generator) > (float)0.5f) {
      clusters.push_back(toMat(objectPoints2D_[i]));
    }
  }
  
  return clusters;
}

///Linear 2D extrapolation/interpolation
/// @param coords vector of 2 points having coordinates with known values.
/// @param values vector of 2 known values.
/// @param newCoord new point having coordinates for extrapolated values.
/// @returns extrapolated value at specified point.
double extrapolate(const std::vector<std::vector<double>> &coords, const std::vector<double> &values, const std::vector<double> &newCoord)
{
    double y;
    double distanceX, distanceY, distance;

    // calculate 2D Euclidean distance
    // prevent division by zero
    if((coords[1][0] - coords[0][0]) == 0.0) {
      distanceX = 0.0;
    }
    else {
      distanceX = std::pow((newCoord[0] - coords[1][0]) / (coords[1][0] - coords[0][0]) + 1.0, 2.0);
    }
    if((coords[1][1] - coords[0][1]) == 0.0) {
      distanceY = 0.0;
    }
    else {
      distanceY = std::pow((newCoord[1] - coords[1][1]) / (coords[1][1] - coords[0][1]) + 1.0, 2.0);
    }

    if((distanceX + distanceY) == 0.0) {
      distance = 0.0;
    }
    else {
      distance = std::sqrt(distanceX + distanceY);
    }

    // x = std::lerp(coords[0][0], coords[1][0], distance);
    // y = std::lerp(coords[0][1], coords[1][1], distance);
    y = std::lerp(values[0], values[1], distance);
    return y;
}

double extrapolateN(const std::vector<std::vector<double>> &coords, const std::vector<double> &values, const std::vector<double> &newCoord)
{
    double y;
    double distanceX, distanceY, distance;

    // calculate 2-Dimensional Euclidean distance
    for (int i = 1; i < coords.size(); i++)
    {
      if((coords[i][0] - coords[i-1][0]) == 0.0) {
        distanceX = 0.0;
      }
      else {
        // distanceX = std::pow((newCoord[0] - coords[i][0]) / (coords[i][0] - coords[i-1][0]) + 1.0, 2.0);
        distanceX = (newCoord[0] - coords[i][0]) / (coords[i][0] - coords[i-1][0]) + 1.0;
      }
      if((coords[i][1] - coords[i-1][1]) == 0.0) {
        distanceY = 0.0;
      }
      else {
        // distanceY = std::pow((newCoord[1] - coords[i][1]) / (coords[i][1] - coords[i-1][1]) + 1.0, 2.0);
        distanceY = (newCoord[1] - coords[i][1]) / (coords[i][1] - coords[i-1][1]) + 1.0;
      }

      if((distanceX + distanceY) == 0.0) {
        distance = 0.0;
      }
      else {
        // distance = std::sqrt(distanceX + distanceY);
        distance = distanceX + distanceY;
      }
      y += std::lerp(values[0], values[i], distance);
    }
    
    y = y / (coords.size()-1);
    return y;
}

// Function to find unique values.
// @returns unique values and vector with indices of unique values
template <template <typename> class Container, 
          typename T1>
std::tuple<Container<T1>, std::vector<int>> unique(const Container<T1> &vec)
{
    using namespace std;
    using namespace cv;

    Container<T1> uniquePoints;
    vector<int> uniqueIndices;
    unordered_set<T1> s;
    for (int i = 0; i < vec.size(); i++) {
      if(s.contains(vec[i])) {
        // duplicateIndices.push_back(i);
        continue;
      }
      else {
        s.insert(vec[i]);
        uniquePoints.push_back(vec[i]);
        uniqueIndices.push_back(i);
      }
    }    
    // uniquePoints.assign(s.begin(),s.end());

    return make_tuple(uniquePoints, uniqueIndices);
}

/// Function to select values at indices.
template <class T>
T select(T &vec, const std::vector<int> &ind) {
  using namespace std;

  T vec2(ind.size());
  for (int i = 0; i < ind.size(); i++) {
    vec2[i] = vec[ind[i]];
  }
  return vec2;
}
// template <class T>
// std::vector<T> select(std::vector<T> &vec, const std::vector<int> &ind) {
//   using namespace std;

//   std::vector<T> vec2(ind.size());
//   for (int i = 0; i < ind.size(); i++) {
//     vec2[i] = vec[ind[i]];
//   }
//   return vec2;
// }

/**
 * @brief Function to remove values(inplace) at indices.
 * 
 * @param vec input/output vector
 * @param ind vector of indices for removal
 */
template <typename T>
void remove(std::vector<T> &vec, const std::vector<int> &ind) {
  using namespace std;
  std::vector<int> indSorted = ind;
  sort(indSorted.begin(), indSorted.end());
  for (int i = 0; i < ind.size(); i++) {
    vec.erase(vec.begin() + indSorted[indSorted.size()-i-1]);
  }
}

/**
 * @brief Function to remove values(inplace) at indices.
 * 
 * @param vec input/output vector
 * @param ind index for removal
 */
 template <typename T>
 void remove(std::vector<T> &vec, int ind) {
  using namespace std;
  vec.erase(vec.begin() + ind);
 }

 /**
  * @brief Function to swap data between selected positions.
  * 
  * @param target input array-like container
  * @param pos1 first position to swap
  * @param pos2 second position to swap
  * @return copy of Container<T1> target swith swapped data.
  */
template <template <typename> class Container, 
          typename T1>
Container<T1> swap(const Container<T1> &target, int pos1, int pos2) {
  Container<T1> result(target.size());
  std::copy(std::begin(target), std::end(target), std::begin(result));
  // std::copy(std::begin(target)+pos1, std::begin(target)+pos1+1, std::begin(result)+pos2);
  // std::copy(std::begin(target)+pos2, std::begin(target)+pos2+1, std::begin(result)+pos1);
  result[pos1] = target[pos2];
  result[pos2] = target[pos1];

  return result;
}

// Function to swap columns
// @returns cloned cv::Mat with swaped col1 and col2
cv::Mat swapCol(const cv::Mat &target, int col1, int col2) {
  cv::Mat result = target.clone();
  // cv::Mat column1 = target.col(col1).clone();
  target.col(col1).copyTo(result.col(col2));
  target.col(col2).copyTo(result.col(col1));
  // column1.copyTo(result.col(col2));
  return result;
}

/// @brief Fills masked region of image with specified value.
/// @param image image to set.
/// @param mask mask (range [0;1]) that specifies pixels to be changed.
/// @param value new value to apply in masked region.
/// @return image with new value in masked region.
cv::Mat fillMasked(const cv::Mat &image, const cv::Mat &mask, float value)
{
  using namespace std;
  using namespace cv;

  Mat image2 = image.clone();
  Mat mask2 = mask.clone();
  mask2 = NormalizeTo_0_1(mask2);
  image2.convertTo(image2, CV_32F);
  mask2.convertTo(mask2, CV_32F);

  std::vector<cv::Mat> channels;
  cv::split(image2, channels);
  for (int i = 0; i < channels.size(); i++)
  {
    for (int r = 0; r < channels[i].rows; r++)
    {
      for (int c = 0; c < channels[i].cols; c++)
      {
        if(mask2.at<float>(r,c) > (float)0.0f) {
          channels[i].at<float>(r,c) = mask2.at<float>(r,c) * value;
        }
      }    
    }    
  }
  merge(channels, image2);
  
  return image2;
}

//TODO: try adding initial map values from previous iters to calibrateCamera ?
/// Find calibration parameters based on pairs of missaligned points[height,width].
/// @param ratio sets how much more points is extrapolated using faster but less accurate method.
/// Ratio=-1 means all points are result of slower method, ratio=2 means half (1/2) of points is result of faster method.
/// @returns 2 maps with x and y coordindates. Maps have only diffrence values compared to reference straight map (refx-mapx).
std::tuple<cv::Mat, cv::Mat> calibrateCamera(const cv::Mat &_referencePoints, const cv::Mat &_imagePoints,
                                              const cv::Size &imageSize, float ratio=-1, bool addCenterPoint=false, 
                                              bool printing=false){
  using namespace std;
  using namespace cv;

  auto start = std::chrono::high_resolution_clock::now();
  Mat referencePoints = _referencePoints; //_referencePoints; swapCol(_referencePoints,0,1);
  Mat imagePoints = _imagePoints; //_imagePoints; swapCol(_imagePoints,0,1);
  referencePoints.convertTo(referencePoints, CV_32F);
  imagePoints.convertTo(imagePoints, CV_32F);
  cv::Size extrapolationSize = imageSize;
  if(ratio >= 2) {
    extrapolationSize = cv::Size((int)round((float)imageSize.width/ratio), (int)round((float)imageSize.height/ratio)); //extrapolationSize / ratio;
    referencePoints = divide(referencePoints, (float)ratio); //referencePoints / ratio;
    imagePoints = divide(imagePoints, (float)ratio); //imagePoints / ratio;
  }
  referencePoints.convertTo(referencePoints, CV_64F);
  imagePoints.convertTo(imagePoints, CV_64F);

  if(addCenterPoint) {
    //distortion at center should be zero
    Mat centerPoint = Mat::ones(1, 2, CV_64F);
    centerPoint.at<double>(0,0) = (double) extrapolationSize.height / 2;
    centerPoint.at<double>(0,1) = (double) extrapolationSize.width / 2;
    //remove more than 1 aready existing center point
    bool found = false;
    Mat pts1, pts2;
    for(int i=0; i<referencePoints.rows; i++) {
      if(isEqual(referencePoints.row(i), centerPoint) || isEqual(imagePoints.row(i), centerPoint)) {
        found = true;
      }
      else {
        pts1.push_back(referencePoints.row(i));
        pts2.push_back(imagePoints.row(i));
      }
    }
    if(!found) {
      pts1.push_back(centerPoint);
      pts2.push_back(centerPoint); 
    }
    else {
      if(printing || VERBOSITY > 1) {
        cout<<"calibrateCamera:INFO:"<<"center point 0,0 was already in input points"<<endl;
      }
    }
    referencePoints = pts1;
    imagePoints = pts2;
  }

  Mat diff1 = subtract(referencePoints.col(0), imagePoints.col(0));
  Mat diff2 = subtract(referencePoints.col(1), imagePoints.col(1));
  Mat diff;
  hconcat(diff1, diff2, diff);
  diff.convertTo(diff, CV_64F);

  Mat map1 = Mat::zeros(extrapolationSize.height, extrapolationSize.width, CV_64F);
  Mat map2 = Mat::zeros(extrapolationSize.height, extrapolationSize.width, CV_64F);

  // extrapolate maps
  vector<double> knownX, knownY, knownDiff1, knownDiff2;
  knownX = toVec(referencePoints.col(0), (double)0.0);
  knownY = toVec(referencePoints.col(1), (double)0.0);
  knownDiff1 = toVec(diff.col(0), (double)0.0);
  knownDiff2 = toVec(diff.col(1), (double)0.0);
  auto indices = Argsort(knownX);
  knownX = Reorder(knownX, indices);
  knownY = Reorder(knownY, indices);
  knownDiff1 = Reorder(knownDiff1, indices);
  knownDiff2 = Reorder(knownDiff2, indices);
  //TODO: interpolators possibly cant extrapolate and fill with 0 when out of range
  _2D::LinearDelaunayTriangleInterpolator<double> interp1; //LinearDelaunayTriangleInterpolator  ThinPlateSplineInterpolator
  _2D::LinearDelaunayTriangleInterpolator<double> interp2; //LinearDelaunayTriangleInterpolator  ThinPlateSplineInterpolator
  interp1.setData(knownX, knownY, knownDiff1);
  interp2.setData(knownX, knownY, knownDiff2);
  for(double w=0.0; w<extrapolationSize.width; w++) {
    for(double h=0.0; h<extrapolationSize.height; h++) {
      double x = interp1(w, h);
      double y = interp2(w, h);
      map1.at<double>(h,w) = (double) x;
      map2.at<double>(h,w) = (double) y;
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

  map1 = patchNaNs(map1, (double)0.0,(double)0.0,(double)0.0);
  map2 = patchNaNs(map2, (double)0.0,(double)0.0,(double)0.0);
  map1.convertTo(map1, CV_32F);
  map2.convertTo(map2, CV_32F);
    // medianBlur(map1, map1, 3);
    // medianBlur(map2, map2, 3);

  double minResult1, maxResult1;
  cv::minMaxLoc(map1, &minResult1, &maxResult1);
  double minResult2, maxResult2;
  cv::minMaxLoc(map2, &minResult2, &maxResult2);
  if(printing || VERBOSITY > 1) {
    std::cout << "calibrateCamera: elapsed time = "<< MicroToSeconds(duration.count()) << " [s]" << std::endl;
    cout << "calibrateCamera: extrapolated map1 - minVal=" << minResult1 << " ; maxVal=" << maxResult1 << endl;
    cout << "calibrateCamera: extrapolated map2 - minVal=" << minResult2 << " ; maxVal=" << maxResult2 << endl;
  }

  // interpolate (if needed) maps to full size using faster method
  float ratio1 = ((float)imageSize.height/(float)map1.rows); 
  float ratio2 = ((float)imageSize.width/(float)map1.cols); //((maxInput2-minInput2) / (maxResult2-minInput2));
  // float ratio1 = (float) (imageSize.height/2.0f) / ((maxResult1-minResult1)/2.0f);
  // float ratio2 = (float) (imageSize.width/2.0f) / ((maxResult2-minResult2)/2.0f);
  if(ratio >= 2) {
    // map1 = multiply(map1, (float)ratio1);
    // map2 = multiply(map2, (float)ratio2);
    map1 = Resize(map1, imageSize.width, imageSize.height, cv::INTER_LANCZOS4);
    map2 = Resize(map2, imageSize.width, imageSize.height, cv::INTER_LANCZOS4);
    // cv::normalize(map1, map1, (double)0, (double)imageSize.width, NORM_MINMAX);
    // cv::normalize(map2, map2, (double)0, (double)imageSize.height, NORM_MINMAX);
  }

  // // correct/remove extrapolated diffrences from ideal maps
  // auto[plainX,plainY] = initMap2D(map2.rows, map1.cols);
  // map1 = add(plainX,map1); //plainX - map1;
  // map2 = add(plainY,map2); //plainY - map2;

  return std::make_tuple(map1, map2);
}

/// @brief Find calibration parameters based on pairs of missaligned points[height,width]
/// @returns 2 maps with x and y coordindates and calculated camera matrix and distortion coeffs. Maps have full coordinate alignment values.
std::tuple<cv::Mat, cv::Mat, cv::Mat, cv::Mat> calibrateCameraOCV( const std::vector<std::vector<cv::Point3f>> &objectPoints,
                                              const std::vector<std::vector<cv::Point2f>> &imagePoints,
                                              const cv::Size &imageSize,
                                              double checkArea=0.7, 
                                              int flags=0,
                                              cv::Mat mtx=cv::Mat::eye(3,3,CV_32F),
                                              cv::Mat distortion = cv::Mat::zeros(14,1,CV_32F) ) {
  using namespace std;
  using namespace cv;
  
  Mat rvecs, tvecs;
  double result = calibrateCamera(objectPoints, imagePoints, imageSize, mtx, distortion, rvecs, tvecs, flags);
  mtx.convertTo(mtx, CV_32F);
  distortion.convertTo(distortion, CV_32F);
  Mat newcameramtx = Mat::eye(3,3,CV_32F); //TODO: test without overwriting matrix to eye?
  Mat newMap1,newMap2;
  int width = (int)(checkArea * (double)imageSize.width);
  int height = (int)(checkArea * (double)imageSize.height);
  cv::Rect *roi = new cv::Rect((imageSize.width-width)/2,(imageSize.height-height)/2, width,height);
  newcameramtx = cv::getOptimalNewCameraMatrix(mtx, distortion, imageSize, 1.0, imageSize, 
                                              roi, true);
  Mat R;
  initUndistortRectifyMap(newcameramtx, distortion, R, newcameramtx , imageSize, CV_32FC1, newMap1, newMap2);
  newMap1.convertTo(newMap1, CV_32F);
  newMap2.convertTo(newMap2, CV_32F);
  return std::make_tuple(newMap1, newMap2, newcameramtx, distortion);
}

// Mirrors selected quarter of image to the rest of image
cv::Mat mirror4way(const std::vector<cv::Mat> &partial_images, int index)
{
  using namespace std;
  using namespace cv;
  
  auto partial_images2 = partial_images;
  Mat mirrored;
  Mat mirrored2;
  int w = 0;
  int h = 0;

  for (int i = 0; i < 4; i++)
  {
    w = max(w, partial_images2[i].cols);
    h = max(h, partial_images2[i].rows);

    if(i == index) {
      continue;
    }
    
    if(i < 2) {
      if(index < 2) {
          cv::flip(partial_images2[index], partial_images2[index], 1);
      }
      else {
          cv::flip(partial_images2[index], partial_images2[index], 0);
      }
    }
    else if(i >= 2) {
      if(index >= 2) {
          cv::flip(partial_images2[index], partial_images2[index], 1);
      }
      else {
          cv::flip(partial_images2[index], partial_images2[index], 0);
      }
    }
  }
  

  hconcat(partial_images[0], partial_images[1], mirrored);
  hconcat(partial_images[2], partial_images[3], mirrored2);
  vconcat(mirrored, mirrored2, mirrored);
  return mirrored;
}

// Find best parameters to match pair of images
std::tuple<cv::Mat, cv::Mat> relativeUndistort(const cv::Mat &refImg, const cv::Mat &distortedImg,
                                              int cameraWidth, int cameraHeight, 
                                              ProgramParams pparams=ProgramParams(), AlignmentParams aparams=AlignmentParams())
{
  using namespace std;
  using namespace cv;
  
  if(VERBOSITY > 0) {
    cout << "RelativeUndistort: Finding and interpolating parameters to align/undistort image" << endl;
  }
  int validSolutions = 0;
  int validSolutions2 = 0;

  bool useCenterMask = false;
  int iter_milestone = 12; //make extra adjustments after reaching it
  double checkArea = aparams.checkArea; //0.7 //0.8 0.9
  double alpha = aparams.alpha; //0.7; //1.0
  int maxIter = aparams.maxIter; //100; //1 // 50; 15
  bool alignCenter = aparams.alignCenter; //false; //false
  auto interp = (cv::InterpolationFlags)pparams.interpolation; //cv::INTER_LANCZOS4; //cv::INTER_LINEAR; cv::INTER_LANCZOS4
  bool warpAlign = aparams.warpAlign; //true;
  int warpIter = aparams.warpIter; //0
  int K = aparams.K; //3; //matchedPoints2.rows/7; //3
  int n_points = aparams.n_points; //1024; //4096 //8192 //512
  float ratio = aparams.ratio; //0.75f; //how many points to keep for alignment //0.5f;
  bool mirroring = aparams.mirroring; //false; //try mirroring best alignment //false

  cv::Mat img2, dist2;
  cv::Mat maskImg, maskDist;
  std::vector<double> mseList;
  std::vector<std::vector<int>> resolutions;
  double mse_ref, mse_dist;
  //use only 1 channel for faster processing
  if(refImg.channels() > 1) {
    cv::cvtColor(refImg.clone(), img2, cv::COLOR_BGR2GRAY);
  }
  if(distortedImg.channels() > 1) {
    cv::cvtColor(distortedImg.clone(), dist2, cv::COLOR_BGR2GRAY);
  }
  // cv::cvtColor(Resize(refImg.clone(), refImg.cols/2, refImg.rows/2), img2, cv::COLOR_BGR2GRAY);
  // cv::cvtColor(Resize(distortedImg.clone(), refImg.cols/2, refImg.rows/2), dist2, cv::COLOR_BGR2GRAY);
  //reduce resolution for faster processing
  if(refImg.rows >= 4000) {
    img2 = Resize(refImg.clone(), refImg.cols/4, refImg.rows/4);
    dist2 = Resize(distortedImg.clone(), refImg.cols/4, refImg.rows/4);
  }
  else if(refImg.rows >= 1024) {
    img2 = Resize(refImg.clone(), refImg.cols/2, refImg.rows/2);
    dist2 = Resize(distortedImg.clone(), refImg.cols/2, refImg.rows/2);
  }
  maskImg = maskFromColor(img2);
  mse_ref = CompareImg(img2, dist2, checkArea);
  mseList.push_back(mse_ref);
  resolutions.push_back({dist2.cols, dist2.rows});
  if(VERBOSITY > 0) {
    cout << "RelativeUndistort: mse of input image before optimizations = " << mse_ref << endl;
    }

  // find optimal parameters
  auto [h, w] = std::make_tuple(dist2.rows, dist2.cols);
  cv::Mat mtx = Mat::eye(3,3,CV_32F);
  cv::Mat distortion= Mat::zeros(14,1,CV_32F);
  cv::Size imageSize;
  imageSize.height=h;
  imageSize.width=w;
  bool found_undistortion = false;
  cv::Mat undistorted;
  auto[map1_ref,map2_ref] = initMap2D(dist2.rows, dist2.cols);
  auto map1 = map1_ref.clone();
  auto map2 = map2_ref.clone();
        
  //Align images
  cv::Mat aligned;
  int width = (int)(checkArea * (double)imageSize.width);
  int height = (int)(checkArea * (double)imageSize.height);
  cv::Rect *roi = new cv::Rect((imageSize.width-width)/2,(imageSize.height-height)/2, width,height);
  Rect imageCenter = *roi;
  Mat maskCenter(imageSize, CV_8UC1, Scalar::all(0));
  maskCenter(imageCenter).setTo(Scalar::all(255));
  if(warpAlign) {
    // TODO: add alignimage mask
    //TODO: add alignimage for map1,map2
    if(!useCenterMask) {
    aligned = AlignImageToImage(img2,dist2, 
                          n_points,1.2f,16,ratio,3, 
                          10.0f, nullptr, BORDER_CONSTANT, warpIter);
    }
    else {
     aligned = AlignImageToImage(img2,dist2, 
                          n_points,1.2f,16,ratio,3, 
                          10.0f, &maskCenter, BORDER_CONSTANT, warpIter);
    }
    mse_dist = CompareImg(img2, aligned, checkArea);
    if(mse_dist > mse_ref) {
      aligned = dist2;
    }
    mseList.push_back(mse_dist);
    resolutions.push_back({aligned.cols, aligned.rows});
  }
  else {
    aligned = dist2;
  }

  // int n_points = 4096; //8192; 1024; 2048;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution((float)0.25f,(float)2.0f);
  std::uniform_real_distribution<float> distribution2((float)0.85f,(float)1.1f);

  if (maxIter > 0)
  {
    dist2 = aligned.clone();
    maskDist = maskFromColor(dist2);  
    Mat maskCombined = multiply(maskImg, maskDist);
    // maskCombined = invertMask(maskCombined);
    maskCombined.convertTo(maskCombined, CV_8U, 255.0);
    mse_dist = CompareImg(img2, dist2, checkArea);
    vector< vector<Point3f> > objectPoints(0, vector<Point3f>());
    vector< vector<Point2f> > imagePoints(0, vector<Point2f>()); // 2d points in new image
    
    for (int i = 0; i < maxIter; i++)
    {
      if(i % iter_milestone == (iter_milestone-1)) {
        objectPoints.clear();
        imagePoints.clear();
      }

      //gather points for finding undistotion
      int gatherIter = min(3,maxIter);
      int listSize = objectPoints.size();
      objectPoints.resize(listSize+gatherIter);
      imagePoints.resize(listSize+gatherIter);
      vector<Mat> points1(gatherIter);
      vector<Mat> points2(gatherIter);
      #pragma omp parallel for
      for (int j = 0; j < gatherIter; j++)
      {
        //Find undistortion for aligned images
        int rand_features =  (float)n_points * distribution(generator);
        float pointsRatio = (float)ratio * distribution2(generator);
        if(VERBOSITY > 1) {
          cout << "calibrateCamera: rand_features=" << rand_features << endl;
        }      
        //detect points for undistortion
        auto[matchedPoints1_,matchedPoints2_] = DetectFeatures(img2, dist2, rand_features, Mat(),Detector::orb, 
                                                                                  1.3f,8,pointsRatio,3,false);
        if((matchedPoints1_.rows < 8) || (matchedPoints2_.rows < 8)) {
          if(VERBOSITY > 1) {
            cout << "RelativeUndistort: skipping aligment with incorrect points" << endl;
          }
          continue;
        }
        matchedPoints1_.convertTo(matchedPoints1_, CV_32F);
        matchedPoints2_.convertTo(matchedPoints2_, CV_32F);
        points1[j] = matchedPoints1_;
        points2[j] = matchedPoints2_;
        objectPoints[listSize+j] = toVecPoint3f(matchedPoints1_);
        imagePoints[listSize+j] = toVecPoint2f(matchedPoints2_);
      }

      Mat matchedPoints1 = points1[0];
      Mat matchedPoints2 = points2[0];
      for (int j = 1; j < gatherIter; j++) {
        vconcat(matchedPoints1, points1[j], matchedPoints1);
        vconcat(matchedPoints2, points2[j], matchedPoints2);
      }
      // i = i+gatherIter-1;

      vector< vector<Point2f> > objectPoints2D(1, vector<Point2f>(objectPoints.back().size()));
      objectPoints2D.at(0) = toVecPoint2f(objectPoints.back());

      //TODO: scale points to match current resolution of image?
      auto average1 = cv::mean(toMat(objectPoints2D[0]))[0];
      auto average2 = cv::mean(toMat(imagePoints.back()))[0];
      //check for bad points
      if((average1 <= 1.0) || (average2 <= 1.0)) {
        if(VERBOSITY > 1) {
          cout << "RelativeUndistort: skipping aligment with incorrect points" << endl;
        }
        continue;
      }
      
      Mat bestLabels,bestLabels2;
      cv::TermCriteria termCriteria(cv::TermCriteria::MAX_ITER + cv::TermCriteria::EPS, 2000, 1e-3);
      Mat centers,centers2;
      // auto flags_clustering = cv::KMEANS_RANDOM_CENTERS; //Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007]
      // double compactness = kmeans(objectPoints[0], K, bestLabels, termCriteria, attempts, flags_clustering, centers);
      // centers = toMat(objectPoints2D[0]); //cluster4way(K, objectPoints2D[0]); toMat(objectPoints2D[0]);  matchedPoints1;
      Mat combined_points = toMat(objectPoints2D[0]);
      hconcat(combined_points, toMat(imagePoints.back()), combined_points);
      combined_points.convertTo(combined_points, CV_32F);
      if(VERBOSITY > 1) {
        cout << "RelativeUndistort: combined_points.size() = " << combined_points.size() << endl; 
      }

      //TODO: test adding oryginal points to clusters_centers 
      // Mat centers_combined = cluster4way(K, toVec2D(combined_points, (float)0.0f));
      Mat centers_combined = getKRandPoints(K, toVec2D(combined_points, (float)0.0f));

      centers = centers_combined.col(0);
      hconcat(centers, centers_combined.col(1), centers);
      centers.convertTo(centers, CV_32F);
      vector<Point2f> center_points = toVecPoint2f(centers);
      // double compactness2 = kmeans(imagePoints[0], K, bestLabels2, termCriteria, attempts, flags_clustering, centers2);
      // centers2 = toMat(imagePoints[0]); //cluster4way(K, imagePoints[0]);  toMat(imagePoints[0]);  matchedPoints2;
      centers2 = centers_combined.col(2);
      hconcat(centers2, centers_combined.col(3), centers2);
      centers2.convertTo(centers2, CV_32F);
      vector<Point2f> center_points2 = toVecPoint2f(centers2);
        //show clusters
        Mat figure = img2.clone();
        if(VERBOSITY > 1) { 
          for(auto p : center_points){
            cv::circle(figure, p, 15, {100, 255, 100}, 5);
          }
          for(auto p2 : center_points2){
            cv::circle(figure, p2, 10, {150, 150, 255}, 3);
          }
          imshow("Detected features - Cluster centers", figure );
          waitKey(5000);
          cv::imwrite("./Detected features - Cluster centers.jpg", figure); //test
        }

      Mat rvecs,tvecs;     
      // auto flags = cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_RATIONAL_MODEL;
      // auto flags = cv::CALIB_ZERO_TANGENT_DIST + cv::CALIB_CB_CLUSTERING;
      // auto flags = cv::CALIB_USE_INTRINSIC_GUESS+cv::CALIB_CB_CLUSTERING+cv::CALIB_FIX_FOCAL_LENGTH+cv::CALIB_RATIONAL_MODEL;
      // auto flags = cv::CALIB_USE_INTRINSIC_GUESS+cv::CALIB_CB_CLUSTERING+cv::CALIB_FIX_FOCAL_LENGTH;
      auto flags = cv::CALIB_ZERO_TANGENT_DIST+cv::CALIB_CB_CLUSTERING+cv::CALIB_FIX_FOCAL_LENGTH;
      // int flags = 0;
      double result = calibrateCamera(objectPoints, imagePoints, imageSize, mtx, distortion, rvecs, tvecs, flags);
      mtx.convertTo(mtx, CV_32F);
      distortion.convertTo(distortion, CV_32F);
      Mat newcameramtx = Mat::eye(3,3,CV_32F); //TODO: test without overwriting matrix to eye?
      // newcameramtx.at<float>(0, 2) = w/2; //imageSize.width/2 - mtx.at<float>(0,2); // w/2;
      // newcameramtx.at<float>(1, 2) = h/2; //imageSize.height/2 - mtx.at<float>(1,2); // h/2;
      // distortion = Mat::zeros(14,1,CV_32F);
      Mat newMap1,newMap2;
      newcameramtx = cv::getOptimalNewCameraMatrix(mtx, distortion, imageSize, 1.0, imageSize, 
                                                   roi, true);
      // Mat R = Mat::eye(3,3,CV_32F);
      Mat R;
      initUndistortRectifyMap(newcameramtx, distortion, R, newcameramtx , imageSize, CV_32FC1, newMap1, newMap2);
      newMap1.convertTo(newMap1, CV_32F);
      newMap2.convertTo(newMap2, CV_32F);

      // // auto[map1,map2] = calibrateCamera(matchedPoints1, matchedPoints2, imageSize);
      float resolutionRatio = min(8.0f, (float)imageSize.height/16);
      // auto[newMap1,newMap2] = calibrateCamera(centers, centers2, imageSize, resolutionRatio, false);

      cv::Mat undistorted;
      undistorted = Undistort(dist2.clone(), newMap1, newMap2, interp);
      if(!useCenterMask) {
        undistorted = AlignImageToImage(img2,undistorted);
      }
      else {
        undistorted = AlignImageToImage(img2, undistorted, 
                          4000,1.5f,5,0.75f,3, 
                          10.0f, &maskCenter);
      }
      double mse = CompareImg(img2, undistorted, checkArea);

      //TODO: test mirroring good part of map to other parts
      if(mirroring) {
        Mat topLeft,topRight,bottomLeft,bottomRight;
        Mat topLeft_newMap1,topRight_newMap1,bottomLeft_newMap1,bottomRight_newMap1;
        Mat topLeft_newMap2,topRight_newMap2,bottomLeft_newMap2,bottomRight_newMap2;
        Point2i topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner;
        topLeftCorner = {0, 0}; //Point2i(0,0); //{0, 0};
        topRightCorner = {0, w/2}; //Point2i(w/2,0); //{0, w/2};
        bottomLeftCorner = {h/2, 0}; //Point2i(0,h/2); //{h/2, 0};
        bottomRightCorner = {h/2, w/2}; //Point2i(w/2,h/2); //{h/2, w/2};
        cv::Rect topLeft_roi = CornersToRect(topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        topLeft = CutImgToCorners(undistorted, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        topLeft_newMap1 = CutImgToCorners(newMap1, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        topLeft_newMap2 = CutImgToCorners(newMap2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        double mse_topLeft = CompareImg(CutImgToCorners(img2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner), topLeft, checkArea);
        topLeftCorner = {0, w/2}; //Point2i(0,0); //{0, w/2};
        topRightCorner = {0, w-1}; //Point2i(w-1,0); //{0, w-1};
        bottomLeftCorner = {h/2, w/2}; //Point2i(0,h/2); //{h/2, w/2};
        bottomRightCorner = {h/2, w-1}; //Point2i(w-1,h/2); //{h/2, w-1};
        cv::Rect topRight_roi = CornersToRect(topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        topRight = CutImgToCorners(undistorted, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        topRight_newMap1 = CutImgToCorners(newMap1, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        topRight_newMap2 = CutImgToCorners(newMap2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        double mse_topRight = CompareImg(CutImgToCorners(img2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner), topRight, checkArea);
        topLeftCorner = {h/2, 0};
        topRightCorner = {h/2, w/2};
        bottomLeftCorner = {h-1, 0};
        bottomRightCorner = {h-1, w/2};
        cv::Rect bottomLeft_roi = CornersToRect(topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        bottomLeft = CutImgToCorners(undistorted, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        bottomLeft_newMap1 = CutImgToCorners(newMap1, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        bottomLeft_newMap2 = CutImgToCorners(newMap2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        double mse_bottomLeft = CompareImg(CutImgToCorners(img2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner), bottomLeft, checkArea);
        topLeftCorner = {h/2, w/2};
        topRightCorner = {h/2, w-1};
        bottomLeftCorner = {h-1, w/2};
        bottomRightCorner = {h-1, w-1};
        cv::Rect bottomRight_roi = CornersToRect(topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        bottomRight = CutImgToCorners(undistorted, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        bottomRight_newMap1 = CutImgToCorners(newMap1, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        bottomRight_newMap2 = CutImgToCorners(newMap2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        double mse_bottomRight = CompareImg(CutImgToCorners(img2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner), bottomRight, checkArea);
        
        vector<double> mse_partial = {mse_topLeft, mse_topRight, mse_bottomLeft, mse_bottomRight};
        vector<Rect> partial_roi = {topLeft_roi, topRight_roi, bottomLeft_roi, bottomRight_roi};
        vector<Mat> partial_newMap1 = {topLeft_newMap1, topRight_newMap1, bottomLeft_newMap1, bottomRight_newMap1};
        vector<Mat> partial_newMap2 = {topLeft_newMap2, topRight_newMap2, bottomLeft_newMap2, bottomRight_newMap2};
        int ind = min_element(mse_partial.begin(), mse_partial.end()) - mse_partial.begin();
        newMap1 = mirror4way(partial_newMap1, ind);
        newMap2 = mirror4way(partial_newMap2, ind);
        newMap1 = MatchResolution(newMap1, imageSize, interp);
        newMap2 = MatchResolution(newMap2, imageSize, interp);
        Mat undistorted2 = Undistort(dist2, newMap1, newMap2, interp);
        double mse2 = CompareImg(img2, undistorted2, checkArea);
        mse = mse2;
      }
          
      double minTest, maxTest;
      if(VERBOSITY > 1) {
        cout << "RelativeUndistort: mse after partial undistortion = " << mse << endl;
        cv::minMaxLoc(newMap1, &minTest, &maxTest);
        cout << "RelativeUndistort: min newMap1 = " << minTest << " | max = " << maxTest << endl;
        cv::minMaxLoc(newMap2, &minTest, &maxTest);
        cout << "RelativeUndistort: min newMap2 = " << minTest << " | max = " << maxTest << endl;
      }
      mseList.push_back(mse);
      resolutions.push_back({img2.cols, img2.rows});
      if(mse <= *min_element(mseList.begin(), mseList.end())) {
        found_undistortion = true;
        validSolutions++;
        map1 = newMap1;
        map2 = newMap2;
        if(VERBOSITY > 1) {
          show(figure,"Detected features - Cluster centers", 5000);
          cv::imwrite("./Detected features - Cluster centers.jpg", figure); //test
        }
        objectPoints.clear();
        imagePoints.clear();
        // break; 
        //TODO: combine mtx,dist params from multiple iters (maybe combine maps) ?
      }

      //second undistortion method
      if( (mse <= *min_element(mseList.begin(), mseList.end())) || (i % iter_milestone == 0) ) {
        auto[matchedPoints1,matchedPoints2] = DetectFeatures(img2, undistorted, 2000, Mat(),Detector::akaze, 
                                                                                 1.1f,8,ratio*1.095f,2,false);
        if((matchedPoints1.rows < 8) || (matchedPoints2.rows < 8)) {
          continue;
        }
        matchedPoints1.convertTo(matchedPoints1, CV_32F);
        matchedPoints2.convertTo(matchedPoints2, CV_32F);
        average1 = cv::mean(matchedPoints1)[0];
        average2 = cv::mean(matchedPoints2)[0];
        //check for bad points
        if((average1 <= 1.0) || (average2 <= 1.0)) {
          continue;
        }
        combined_points = matchedPoints1.colRange(0,2);
        hconcat(combined_points, matchedPoints2.colRange(0,2), combined_points);
        combined_points.convertTo(combined_points, CV_32F);
        // centers_combined = cluster4way(16, toVec2D(combined_points, (float)0.0f), Size2f(img2.cols,img2.rows));
        centers_combined = getKRandPoints(K, toVec2D(combined_points, (float)0.0f));
        centers = centers_combined.col(0);
        hconcat(centers, centers_combined.col(1), centers);
        centers2 = centers_combined.col(2);
        hconcat(centers2, centers_combined.col(3), centers2);
        // auto flags_clustering = cv::KMEANS_PP_CENTERS; //Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007]
        // double compactness = kmeans(matchedPoints1, 64, bestLabels, termCriteria, 200, flags_clustering, centers);
        // double compactness2 = kmeans(matchedPoints2, 64, bestLabels2, termCriteria, 200, flags_clustering, centers2);
        centers.convertTo(centers, CV_32F);        
        centers2.convertTo(centers2, CV_32F);
        // auto[newMap21,newMap22] = calibrateCamera(centers, centers2, imageSize, resolutionRatio, false);
        vector< vector<Point3f> > objectPoints2(0, vector<Point3f>());
        vector< vector<Point2f> > imagePoints2(0, vector<Point2f>()); // 2d points in new image
        objectPoints2.push_back(toVecPoint3f(centers));
        imagePoints2.push_back(toVecPoint2f(centers2));
        auto[newMap21,newMap22, mtx2,distortion2] = calibrateCameraOCV(objectPoints2, imagePoints2, imageSize, 
                                                                              checkArea, flags, mtx, distortion);
        newMap1 = newMap21;
        newMap2 = newMap22;
        Mat undistorted2 = Undistort(undistorted.clone(), 
                                    subtract(add(map1,newMap1), map1_ref), 
                                    subtract(add(map2,newMap2), map2_ref), 
                                    interp);
        double mse3 = CompareImg(img2, undistorted2, checkArea);
        Mat test1 = Undistort(dist2.clone(), map1, map2, interp);
        double test_mse1 = CompareImg(img2, test1, checkArea);
          // Mat test2 = Undistort(undistorted.clone(), newMap1, newMap2, interp);
          // Mat test3 = Undistort(undistorted.clone(),add(map1,newMap1), add(map2,newMap2), interp);
          // Mat test4 = Undistort(undistorted.clone(),subtract(map1,newMap1), subtract(map2,newMap2), interp);
          // Mat test5 = Undistort(undistorted.clone(),subtract(add(map1,newMap1), map1_ref), subtract(add(map2,newMap2), map2_ref), interp);
          // double test_mse2 = CompareImg(img2, test2, checkArea);
          // double test_mse3 = CompareImg(img2, test3, checkArea);
          // double test_mse4 = CompareImg(img2, test4, checkArea);
          // double test_mse5 = CompareImg(img2, test5, checkArea);
          if(VERBOSITY > 1) {
            Mat figure = img2.clone();
            for(auto p : toVec2D(centers, (float)0.0f)){
              cv::circle(figure, Point2f(p[0],p[1]), 15, {100, 255, 100}, 3);
            }
            for(auto p2 : toVec2D(centers2, (float)0.0f)){
              cv::circle(figure, Point2f(p2[0],p2[1]), 10, {150, 150, 255}, 2);
            }
            show(figure,"Detected features - Cluster centers - method2", 5000);
            cv::imwrite("./Detected features - Cluster centers - method2.jpg", figure); //test
            double minTest, maxTest;
            cv::minMaxLoc(newMap1, &minTest, &maxTest);
            cout << "RelativeUndistort: newMap1: minVal=" << minTest << " ; maxVal=" << maxTest << endl;
          }

        if(mse3 < test_mse1) {
          validSolutions2++;
          map1 = subtract(add(map1,newMap1), map1_ref);
          map2 = subtract(add(map2,newMap2), map2_ref);
        }
      }
    }
  }

  //scale maps to match input resolution
  float ratio1 = ((float)refImg.rows/(float)map1.rows); 
  float ratio2 = ((float)refImg.cols/(float)map1.cols); 
  if((ratio1 != 1.0f) || (ratio2 != 1.0f)) {
    map1 = multiply(map1, (float)ratio1);
    map2 = multiply(map2, (float)ratio2);
    map1 = Resize(map1, refImg.cols, refImg.rows, interp);
    map2 = Resize(map2, refImg.cols, refImg.rows, interp);
  }

  if(VERBOSITY > 1) {
    cv::imwrite("./calibrateCamera"+string("_map1")+".jpg", map1);
    cv::imwrite("./calibrateCamera"+string("_map2")+".jpg", map2);
    SaveToCSV(mseList, "./RelativeUndistort_mseList.csv");
  }

  // Apply undistortion for final image
  // aligned = distortedImg.clone();
  maskCenter = Resize(maskCenter, refImg.cols, refImg.rows, INTER_NEAREST);
  if(warpAlign) {
    // TODO: add alignimage mask
    if(!useCenterMask) {
      aligned = AlignImageToImage(refImg.clone(), distortedImg.clone(), 
                          n_points,1.2f,16,ratio,3, 
                          10.0f, nullptr, BORDER_CONSTANT, warpIter);
    }
    else {
      aligned = AlignImageToImage(refImg.clone(), distortedImg.clone(), 
                          n_points,1.2f,16,ratio,3, 
                          10.0f, 
                          &maskCenter, BORDER_CONSTANT, warpIter);
    }
  }
  undistorted = Undistort(aligned, map1, map2, interp); // cv::remap(aligned, undistorted, map1, map2, interp);
  // maskDist = maskFromColor(undistorted);
  maskDist = add(maskFromColor(undistorted), invertMask(maskFromColor(refImg)));
  cv::threshold(maskDist,	maskDist, 0.0, 1.0, THRESH_BINARY);
  maskDist.convertTo(maskDist, CV_32F);

  maskDist.convertTo(maskDist, CV_8U, 255.0);
  undistorted = AlignImageToImage(refImg,undistorted, 
                      n_points,1.1f,5,ratio,2, 
                      5.0f, &maskDist, BORDER_CONSTANT, warpIter,
                      false);
  cv::Mat mask;
  cv::Mat distortedImgGray;
  cv::cvtColor(aligned.clone(), distortedImgGray, cv::COLOR_BGR2GRAY);
  // TODO: fix masking for new undistortion algorithm

  bool badUndistortion = false;
  cv::Mat undistoredGray;
  cv::cvtColor(undistorted.clone(), undistoredGray, cv::COLOR_BGR2GRAY);
  mask = maskFromChange(distortedImgGray, undistoredGray, pparams.radicalChangeRatio);
  double counterNonZero = cv::countNonZero(mask);

  if(VERBOSITY > 1) {
    // cout<<"  RelativeUndistort: sum(mask)[0] "<<cv::sum(mask)[0]<<endl;
    cout << "  RelativeUndistort: countNonZero " << counterNonZero << endl;
    cout << "  RelativeUndistort: countNonZero/pixels " << counterNonZero / (undistoredGray.rows * undistoredGray.cols) << endl;
    // cout<<"  RelativeUndistort: sum(undistoredGray)[0] "<<cv::sum(undistoredGray)[0] <<endl; 
  }
  if (counterNonZero > (undistoredGray.rows * undistoredGray.cols * 0.5))
  {
    badUndistortion = false;
  }
  else
  {
    badUndistortion = true;
    if(VERBOSITY > 0) {
      cout << " RelativeUndistort: Warning: bad undistortion of image" << endl;
    }
  }

  double mse_final = CompareImg(refImg, undistorted, checkArea);

  if(VERBOSITY > 0) {
    if(!found_undistortion) {
      cout << "RelativeUndistort: Warning: failed to undistort image " << endl;
    }
    else {
      cout << "RelativeUndistort: Number of found undistortion solutions="<<validSolutions<<" | number of aditional solutions="<<validSolutions2<< endl;
    }
    cout << "RelativeUndistort: final mse of undistored image = " << mse_final <<endl;
  }

  return std::make_tuple(undistorted, mask);
}


//* STACKING *
float sharpness(const cv::Mat& image) {
  using namespace std;
  using namespace cv;

  cv::Mat image2;
  if (image.channels() > 1) { 
      cv::cvtColor(image, image2, cv::COLOR_BGR2GRAY); // cv::cvtColor(image, image2, cv::COLOR_BGR2HSV);

  } else {
      image2 = image;
  }

  int height = std::ceil(image.rows / 2.0);
  int width = std::ceil(image.cols / 2.0);
  std::vector<int> rows, cols;
  for (int i = 0; i < 2; i++) {
      rows.push_back(std::round(i * (image.rows - height)));
      cols.push_back(std::round(i * (image.cols - width)));
  }

  std::vector<std::vector<float>> sharpnessPerRegion;
  for (int row : rows) {
      std::vector<float> sharpnessPerCol;
      for (int col : cols) {
          cv::Rect roi(col, row, width, height);
          cv::Mat region = image2(roi);
          region.convertTo(region, CV_32F);
          sharpnessPerCol.push_back(variance(toValarray(region)));
      }
      sharpnessPerRegion.push_back(sharpnessPerCol);
  }

  float meanSharpness = 0.0;
  for (const auto& row : sharpnessPerRegion) {
      for (float sharpness : row) {
          meanSharpness += sharpness;
      }
  }
  return meanSharpness / (rows.size() * cols.size());
}

cv::Mat sharpnessOfRegions(const cv::Mat& image, float patternSize, int patternN) {
  int height, width;
  if (patternSize < 1.0f) {
      height = std::max(int(image.rows * patternSize), 3);
      width = std::max(int(image.cols * patternSize), 3);
  } else {
      height = std::max(int(patternSize), 3);
      width = std::max(int(patternSize), 3);
  }

  std::vector<int> rows(patternN);
  std::vector<int> cols(patternN);
  for (int i = 0; i < patternN; ++i) {
      rows[i] = std::round(height + i * (image.rows - 2 * height) / (patternN - 1));
      cols[i] = std::round(width + i * (image.cols - 2 * width) / (patternN - 1));
  }

  std::vector<std::vector<float>> sharpnessPerRegion;
  for (int row : rows) {
      std::vector<float> sharpnessPerCol;
      for (int col : cols) {
          cv::Mat region = image(cv::Rect(col, row, width, height));
          sharpnessPerCol.push_back(sharpness(region));
      }
      sharpnessPerRegion.push_back(sharpnessPerCol);
  }

  cv::Mat sharpnessPerRegion_lowRes = toMat(sharpnessPerRegion);
  cv::Mat sharpnessPerRegion_highRes;
  cv::resize(sharpnessPerRegion_lowRes, sharpnessPerRegion_highRes, cv::Size(image.cols,image.rows));

  return sharpnessPerRegion_highRes;
}

std::pair<cv::Point, int> closestPoint(std::vector<cv::Point> pointsXY, cv::Point pointXY, int priority=-1) {
    // priority defines which axis is more important when calculating distance
    std::vector<float> priorityRatio;
    if (priority == -1) {
        priorityRatio = {(float)1.0, (float)1.0};
    } else if (priority == 0) {
        priorityRatio = {(float)1.4, (float)1.0};
    } else if (priority == 1) {
        priorityRatio = {(float)1.0, (float)1.4};
    }

    std::vector<float> distances(pointsXY.size());
    for (int i = 0; i < pointsXY.size(); i++) {
        distances[i] = std::abs(pointsXY[i].x - pointXY.x) * priorityRatio[0] + std::abs(pointsXY[i].y - pointXY.y) * priorityRatio[1];
    }

    int minDistanceIndex = std::distance(std::begin(distances), std::min_element(std::begin(distances), std::end(distances)));
    return std::make_pair(pointsXY[minDistanceIndex], minDistanceIndex);
}

template<typename T>
std::pair<std::vector<T>, int> closestPoint(std::vector<std::vector<T>> pointsXY, std::vector<T> pointXY, int priority=-1) {
    // priority defines which axis is more important when calculating distance
    std::vector<T> priorityRatio;
    if (priority == -1) {
        priorityRatio = {1.0, 1.0};
    } else if (priority == 0) {
        priorityRatio = {1.4, 1.0};
    } else if (priority == 1) {
        priorityRatio = {1.0, 1.4};
    }

    std::vector<T> distances(pointsXY.size());
    for (int i = 0; i < pointsXY.size(); i++) {
        distances[i] = std::abs(pointsXY[i][0] - pointXY[0]) * priorityRatio[0] + std::abs(pointsXY[i][1] - pointXY[1]) * priorityRatio[1];
    }

    int minDistanceIndex = std::distance(std::begin(distances), std::min_element(std::begin(distances), std::end(distances)));
    return std::make_pair(pointsXY[minDistanceIndex], minDistanceIndex);
}

// template<typename T, typename T2>
/// @brief Finds closest value.
/// @param points vector of possible points to search
/// @param x reference value
/// @return closest value to @param x from @param points and index of that value.
template <template <typename> class Container, 
          typename T1, typename T2 >
std::pair<T2, int> closestValue(const Container<T1>& points, T2 x) {

    std::vector<T2> distances(points.size());
    for (int i = 0; i < points.size(); i++) {
      distances[i] = std::abs((T2)(points[i]) - x);
    }

    int minDistanceIndex = std::distance(std::begin(distances), std::min_element(std::begin(distances), std::end(distances)));
    return std::make_pair((T2)points[minDistanceIndex], minDistanceIndex);
}

/// @brief Combines multiple images into single image with maximized sharpness.
/// @param images vector of images to be stacked
/// @param base_id optional which image is treated as base (by default first img is base)
/// @param imagesMasks optional vector of image masks
/// @return single combined image with 3 channels
cv::Mat stackImages( const std::vector<cv::Mat>& images, int base_id=0, const std::vector<cv::Mat>& imagesMasks=std::vector<cv::Mat>(), 
                      ProgramParams pparams=ProgramParams(), StackingParams sparams=StackingParams() ) {
    using namespace std;
    using namespace cv;
    
    int patternN = sparams.patternN;
    int patternSize = sparams.patternSize;
    double minImgCoef = sparams.minImgCoef;
    double baseImgCoef = sparams.baseImgCoef;
    float coef_sharpness = sparams.coef_sharpness;
    float coef_similarity = sparams.coef_similarity;
    double scale = sparams.comparison_scale;
    int blur_size = sparams.blur_size;
    int interpolation = pparams.interpolation;
    float radicalChangeRatio = pparams.radicalChangeRatio;

    std::vector<cv::Mat> images2(images.size());
    for (int i = 0; i < images.size(); i++) {
        // images2[i] = images[i].clone();
      images2[i] = limitChange(images[base_id], images[i], radicalChangeRatio);
    }
    std::vector<Mat> masks2 = imagesMasks;
    if(masks2.size()<images.size()) {
      Mat baseValue = Mat::ones(images[base_id].rows,images[base_id].cols, CV_32F);
      masks2.insert(masks2.begin()+base_id, baseValue);
    }

    std::vector<Mat> sharpnessPerImage;//Mat sharpnessPerImage;
    std::vector<float> averageSharpnessPerImage;
    Mat sharpnessRefImg;
        auto start = std::chrono::high_resolution_clock::now();
    #pragma omp parallel for
    for (int i = 0; i < images2.size(); i++) {
        if(i==base_id) {
            sharpnessRefImg = sharpnessOfRegions(images2[base_id], patternSize, patternN);
            sharpnessRefImg.convertTo(sharpnessRefImg, CV_32F);
            sharpnessPerImage.push_back(sharpnessRefImg);
            averageSharpnessPerImage.push_back(cv::mean(sharpnessRefImg)[0]);
            continue;
        }
        cv::Mat img = images2[i];
        auto sharpnesPerRegion = sharpnessOfRegions(img, patternSize, patternN);
        sharpnesPerRegion.convertTo(sharpnesPerRegion, CV_32F);
        sharpnesPerRegion = multiply(sharpnesPerRegion, masks2[i]);
        sharpnessPerImage.push_back(sharpnesPerRegion);
        averageSharpnessPerImage.push_back(cv::mean(sharpnesPerRegion)[0]);
    }
    if(VERBOSITY > 1) {
        auto stop = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
        std::cout << "StackImages: estimating sharpness of images "<< " | elapsed time = "<< MicroToSeconds(duration.count()) << " [s]" << std::endl;  
    }
    Mat sharpnessAll = to3dMat(sharpnessPerImage);
    sharpnessPerImage.clear();

    std::vector<cv::Mat> diffrencePerImage(images2.size());
    std::vector<float> msePerImage(images2.size());
    cv::Mat baseImgGray;
    cv::cvtColor(images2[base_id].clone(), baseImgGray, cv::COLOR_BGR2GRAY);
    #pragma omp parallel for
    for (int i = 0; i < images2.size(); i++) {
        if(i==base_id) {
          Mat baseValue = Mat(baseImgGray.rows, baseImgGray.cols, CV_32F, (float)0.0);
          diffrencePerImage[i] = baseValue;
          msePerImage[i] = (float)0.0;
          continue;
        }
        cv::Mat inputImg = images2[i].clone();
        cv::Mat ImgGray;
        cv::cvtColor(inputImg, ImgGray, cv::COLOR_BGR2GRAY);
        cv::Mat baseImgGray2 = baseImgGray;
        cv::Mat ImgGray2 = ImgGray;
        if(scale < 1.0) {
          int scaledW = (int)(scale * (double)baseImgGray.size().width);
          int scaledH = (int)(scale * (double)baseImgGray.size().height);
          baseImgGray2 = Resize(baseImgGray, scaledW, scaledH, cv::INTER_AREA);
          ImgGray2 = Resize(ImgGray, scaledW, scaledH);
        }
        cv::Mat diffrences = cv::abs(baseImgGray2 - ImgGray2);
        diffrences = Resize(diffrences, baseImgGray.size().width, baseImgGray.size().height);
        diffrences.convertTo(diffrences, CV_32F);
        // diffrences = multiply(diffrences, masks2[i]);
        diffrencePerImage[i] = diffrences;
        float mse = CompareImg(baseImgGray2, ImgGray2, 0.8);
        msePerImage[i] = mse;
    }
   
    // normalize each point for all photos
    Mat diffrenceAll = to3dMat(diffrencePerImage);
    Mat coeffSum2d = Sum(diffrencePerImage);
    diffrencePerImage.clear();
    diffrenceAll = divide(diffrenceAll, coeffSum2d);
    vector<Mat> temp = toVecMat(sharpnessAll);    
    coeffSum2d = Sum(temp);
    sharpnessAll = divide(sharpnessAll, coeffSum2d);
    sharpnessAll = patchNaNs(sharpnessAll, (float)1.0f / images2.size(), (float)0.0f, (float)1.0f);
    diffrenceAll = patchNaNs(diffrenceAll, (float)1.0f, (float)0.0f, (float)1.0f); //cv::patchNaNs(diffrenceAll, (float)1.0f);
    valarray<float> temp2 = toValarray(NormalizeTo_0_1(msePerImage)) + 1.0f;
    msePerImage = toVec(temp2);
    float minVal = 0.01;
    std::vector<cv::Mat> similarityPerImage(images.size());
    #pragma omp parallel for
    for (int i = 0; i<images.size(); i++) {
      if((i==base_id) && (baseImgCoef > 0.0)) {
        Mat baseValue = Mat(diffrenceAll.rows, diffrenceAll.cols, CV_32F, (float)baseImgCoef);
        similarityPerImage[base_id] = baseValue;
        continue;
    }
        Mat diffImg;
        extractChannel(diffrenceAll, diffImg, i);
        diffImg = divide(1.0f, add(diffImg, minVal));
        diffImg.convertTo(diffImg, CV_32F);
        diffImg = multiply(diffImg, diffImg); //enhance diffrences between good and bad image regions
        similarityPerImage[i] = diffImg;
    }
    diffrenceAll.release();

    //fix wrong values
    #pragma omp parallel for
    for (int i=0;i<similarityPerImage.size();i++) {
      auto coeff = similarityPerImage[i];
      if(i == base_id) {
        coeff = patchNaNs(coeff, (float)1.0f, (float)1.0f, (float)1.0f);
      }
      else {
        coeff = patchNaNs(coeff, (float)1.0f/similarityPerImage.size());
      }
      similarityPerImage[i] = coeff;
    }
    //scale values so that sum of coeffs per pixel is 1.0
    coeffSum2d = Sum(similarityPerImage);
    #pragma omp parallel for
    for (int i=0;i<similarityPerImage.size();i++) {
      similarityPerImage[i] = divide(similarityPerImage[i], coeffSum2d);
    }

    coeffSum2d = Sum(similarityPerImage);
    similarityPerImage = toVecMat(divide(to3dMat(similarityPerImage), coeffSum2d));
    // similarityPerImage = NormalizeTo_0_1(similarityPerImage, 2);
    similarityPerImage = toVecMat(patchNaNs(to3dMat(similarityPerImage), (float)max(similarityPerImage), (float)0.0f, (float)1.0f));

    std::vector<cv::Mat> imageCoeffs(images2.size());
    #pragma omp parallel for
    for (int i = 0; i < images2.size(); i++) {
        Mat sharpnessImg;
        extractChannel(sharpnessAll, sharpnessImg, i);
        imageCoeffs[i] = multiply(sharpnessImg, coef_sharpness); //coef_sharpness * sharpnessImg;
        imageCoeffs[i] = add(imageCoeffs[i], (float)minImgCoef);
    }
    // imageCoeffs[0] = add(similarityPerImage[0], coef_similarity); //coef_similarity * cv::Mat::ones(imageCoeffs[i].rows, 1, CV_32F);
    sharpnessAll.release();

    #pragma omp parallel for
    for (int i = 0; i < images2.size(); i++) {        
        imageCoeffs[i] = add(imageCoeffs[i], multiply(similarityPerImage[i], coef_similarity));
    }
    similarityPerImage.clear();
    coeffSum2d = Sum(imageCoeffs);

    //apply mask to image coeffs
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      if(i != base_id) {
        // Mat mask_inverted = cv::Mat::ones(masks2[0].rows, masks2[0].cols, CV_32F) - masks2[i];
        // imageCoeffs[i] = FillMasked(imageCoeffs[i], mask_inverted, (float)0.0f);
        imageCoeffs[i] = multiply(imageCoeffs[i], masks2[i]);
      }
    }

    //sum of coeffs per pixel should be 1.0
    coeffSum2d = Sum(imageCoeffs);
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      auto coeff = imageCoeffs[i];
      coeff = divide(coeff, coeffSum2d);
      imageCoeffs[i] = coeff;
    }

    // set worst coeffs to 0  //TODO:find min pixels in multichannel image
    // auto replaced = Replace(to3dMat(imageCoeffs), (float)0.0f, ImgReplaceMode::minimum);
    // imageCoeffs = toVecMat(replaced);

    for (int i=0;i<imageCoeffs.size();i++) {
        auto coeff = imageCoeffs[i];
      if(i == base_id) {
        coeff = patchNaNs(coeff, (float)1.0f, (float)1.0f, (float)1.0f);
        // imageCoeffs[base_id] = FillMasked(imageCoeffs[base_id], masks2[i], (float)1.0f);
      }
      else {
        coeff = patchNaNs(coeff, (float)1.0f/images2.size(), (float)0.0f, (float)1.0f);
      }
      coeff = multiply(coeff, masks2[i]);      
      imageCoeffs[i] = coeff;
    }

    //increase diffrences between coeffs
    for (int i=0;i<imageCoeffs.size();i++) {
      imageCoeffs[i] = multiply(imageCoeffs[i], imageCoeffs[i]); //increase diffrences between images
    }

    //make smooth transition between pixels
    for (int i=0;i<imageCoeffs.size();i++) {
      auto coeff = imageCoeffs[i];
      // cv::medianBlur(coeff, coeff, 3);
      cv::blur(coeff, coeff, cv::Size(blur_size, blur_size));
      imageCoeffs[i] = coeff;
    }

    //apply masks to coeffs
    for (int i=0;i<imageCoeffs.size();i++) {
      imageCoeffs[i] = multiply(imageCoeffs[i], masks2[i]);      
    }

    coeffSum2d = Sum(imageCoeffs);
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      auto coeff = imageCoeffs[i];
      coeff = divide(coeff, coeffSum2d); //cv::divide(coeff, coeffSum2d, coeff);
      if(i == base_id) {
        coeff = patchNaNs(coeff, (float)1.0f, (float)1.0f, (float)1.0f);
        // imageCoeffs[base_id] = FillMasked(imageCoeffs[base_id], masks2[i], (float)1.0f);
      }
      else {
        coeff = patchNaNs(coeff, (float)0.0f, (float)0.0f, (float)0.0f);
      }
      imageCoeffs[i] = coeff;
    }

    // //restore base img pixels for masked/unknown regions of images
    // cv::Mat mask = cv::Mat::ones(masks2[0].rows,masks2[0].cols, CV_32F);
    // for (int i = 0; i < masks2.size(); i++) {
    //   mask = multiply(mask, masks2[i]);
    //   Mat borderFill = FillMasked(imageCoeffs[base_id], subtract(Mat::ones(mask.rows,mask.cols, CV_32F), masks2[i]), (float)1.0f/images2.size());
    //   imageCoeffs[base_id] = add(imageCoeffs[base_id], borderFill);
    //   imageCoeffs[i] = multiply(imageCoeffs[i], masks2[i]);
    // }
    // imageCoeffs[base_id] = divide(imageCoeffs[base_id], images2.size());

    coeffSum2d = Sum(imageCoeffs);
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      auto coeff = imageCoeffs[i];
      coeff = divide(coeff,coeffSum2d); //ssum of coeffs for given pixel should be 1.0
      imageCoeffs[i] = coeff;
    }

    double minTest, maxTest;

    Mat stack = Mat::zeros(images2[0].rows, images2[0].cols, CV_32F);
    for (int i = 0; i < images2.size(); i++) {
        cv::Mat img2;
        images2[i].convertTo(img2, CV_32F);
        img2 = multiply(img2, imageCoeffs[i]);
        stack = add(stack,img2);
        cv::minMaxLoc(stack, &minTest, &maxTest);
        // if(VERBOSITY > 1) {
        //   cout << "StackImages: forloop: minVal=" << minTest << " ; maxVal=" << maxTest << endl;
        // }
    }
    imageCoeffs.clear();
    stack.convertTo(stack, CV_32F, 255.0/maxTest);

    cv::Mat sharpnesPerRegion = sharpnessOfRegions(stack, patternSize, patternN);
    cv::Mat sharpnessDiff = sharpnesPerRegion - sharpnessRefImg;
    sharpnessDiff.convertTo(sharpnessDiff, CV_32F);
    double betterSharp = cv::sum(sharpnessDiff >= 0.0)[0] / (stack.rows * stack.cols);
    sharpnesPerRegion.release();
        cv::minMaxLoc(sharpnessDiff, &minTest, &maxTest);
        if(VERBOSITY > 1) {
          cout << "StackImages: stacked - sharpnessDiff: minVal=" << minTest << " ; maxVal=" << maxTest << endl;
        }
    
    //Restore oryginal pixels for regions with reduced sharpnes
    cv::Mat badIndices;
    cv::Mat goodIndices;
    // double res = cv::threshold(sharpnessDiff, goodIndices, 0.0, 1.0, THRESH_BINARY); //values 0 or greater are good
    cv::inRange(sharpnessDiff, 0.0f, numeric_limits<float>::infinity(), goodIndices);
    goodIndices.convertTo(goodIndices, CV_32F, 1.0/255.0);
    Mat radicalChange = maskFromChange(images2[base_id], stack, radicalChangeRatio);
    goodIndices = multiply(goodIndices, radicalChange);
      goodIndices.convertTo(goodIndices, CV_8U, 255.0);
      Mat element = getStructuringElement(MorphShapes::MORPH_ELLIPSE,
                                Size(2*1+1, 2*1+1),
                                Point(1, 1));
      erode(goodIndices, goodIndices, element);
      goodIndices.convertTo(goodIndices, CV_32F, 1.0/255.0);
    // double res2 = cv::threshold(sharpnessDiff,	badIndices, 0.0, 255.0, THRESH_BINARY_INV);
    badIndices = subtract(Mat::ones(goodIndices.rows,goodIndices.cols, CV_32F), goodIndices);
    // double res2 = cv::threshold(badIndices,	badIndices, 0.0, 1.0, THRESH_BINARY);
    // cv::inRange(badIndices, 1.0f, numeric_limits<float>::infinity(), badIndices);
    // cv::medianBlur(goodIndices, goodIndices, blur_size);
    // cv::medianBlur(badIndices, badIndices, blur_size);
    // cv::blur(goodIndices, goodIndices, cv::Size(blur_size, blur_size));
    cv::blur(badIndices, badIndices, cv::Size(blur_size, blur_size));
    coeffSum2d = add(goodIndices, badIndices);
    goodIndices = divide(goodIndices, coeffSum2d);
    badIndices = divide(badIndices, coeffSum2d);
    badIndices = patchNaNs(badIndices, (float)1.0f, (float)1.0f, (float)1.0f);
    goodIndices = patchNaNs(goodIndices, (float)0.0f, (float)0.0f, (float)0.0f);

      if(VERBOSITY > 0)  {
        auto testSum = cv::sum(goodIndices)[0];
        cout<<"StackImages: TEST: number of good pixels = "<<testSum/(badIndices.rows*badIndices.cols)<<endl;
        testSum = cv::sum(badIndices)[0]; //
        cout<<"StackImages: TEST: number of bad pixels = "<<testSum/(badIndices.rows*badIndices.cols)<<endl;
      }
    cv::Mat stackRestoring, baseImgRestoring;
    stackRestoring = stack;
    baseImgRestoring = images[base_id].clone();
    double minStack, maxStack, minBaseimg, maxBaseimg;
    cv::minMaxLoc(stackRestoring, &minStack, &maxStack);
    cv::minMaxLoc(baseImgRestoring, &minBaseimg, &maxBaseimg);
    // stackRestoring = multiply(add(NormalizeTo_0_1(stackRestoring),minStack), maxStack);
    // baseImgRestoring = multiply(add(NormalizeTo_0_1(baseImgRestoring),minBaseimg), maxBaseimg); //NormalizeTo_0_1(baseImgRestoring);
    stackRestoring.convertTo(stackRestoring, CV_32F);
    baseImgRestoring.convertTo(baseImgRestoring, CV_32F);
    // cv::cvtColor(stackRestoring, stackRestoring, cv::COLOR_BGR2Lab);
    // cv::cvtColor(baseImgRestoring, baseImgRestoring, cv::COLOR_BGR2Lab);
    std::vector<cv::Mat> stackChannels, baseImgChannels;
    cv::split(stackRestoring, stackChannels);
    cv::split(baseImgRestoring, baseImgChannels);
    stackChannels[0] = add(multiply(stackChannels[0], goodIndices), multiply(baseImgChannels[0], badIndices)); //baseImgChannels[0]; // stackChannels[0].setTo(baseImgChannels[0], badIndices);
    stackChannels[1] = add(multiply(stackChannels[1], goodIndices), multiply(baseImgChannels[1], badIndices)); //baseImgChannels[1]; //stackChannels[1].setTo(baseImgChannels[1], badIndices);
    stackChannels[2] = add(multiply(stackChannels[2], goodIndices), multiply(baseImgChannels[2], badIndices)); //baseImgChannels[2]; //(1.0f * stackChannels[2] + 2.0f * baseImgChannels[2]) / 3.0f; //stackChannels[2] = (1.0 * stackChannels[2] + 2.0 * baseImgChannels[2]) / 3.0;
    cv::merge(stackChannels, stack);
    stackChannels.clear();
    baseImgChannels.clear();

    //final sharpnes diffrence
    sharpnesPerRegion = sharpnessOfRegions(stack, patternSize, patternN);
    sharpnessDiff = sharpnesPerRegion - sharpnessRefImg;
    // res = cv::threshold(sharpnessDiff, goodIndices, 0.0, 1.0, THRESH_BINARY);
    cv::inRange(sharpnessDiff, 0.0f, numeric_limits<float>::infinity(), goodIndices);
    goodIndices.convertTo(goodIndices, CV_32F, 1.0/255.0);
    betterSharp = cv::sum(goodIndices)[0] / (stack.rows * stack.cols);
    if(VERBOSITY > 0) {
      cout<<"StackImages: TEST: number of good pixels after fix = "<<betterSharp<<endl;
    }
    
    // //Cut unknown regions from final image
    // // cv::findNonZero(mask, rowsOfInterest, colsOfInterest);
    // vector<Point> points;
    // // std::vector<int> rowsOfInterest, colsOfInterest;
    // cv::findNonZero(mask, points);
    // // std::vector<int> rowsOfInterest(toVec(points.col(0), (int)0));
    // // std::vector<int> colsOfInterest(toVec(points.col(1), (int)0));

    // cv::Point topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner;
    // int ind;
    // tie(topLeftCorner,ind) = ClosestPoint(points, cv::Point(0, 0));
    // tie(topRightCorner,ind) = ClosestPoint(points, cv::Point(0, mask.cols));
    // tie(bottomLeftCorner,ind) = ClosestPoint(points, cv::Point(mask.rows, 0));
    // tie(bottomRightCorner,ind) = ClosestPoint(points, cv::Point(mask.rows, mask.cols));
    // list<cv::Point> corners = {topLeftCorner,topRightCorner,bottomLeftCorner,bottomRightCorner};
    // // cout<< "StackImages: corners of image with known values [topLeftCorner,topRightCorner,bottomLeftCorner,bottomRightCorner] = " << to_string(corners) << endl;
    // stack = CutImgToCorners(stack, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner); //if baseimg is not changed then there's no point of cutting it
    
    // stack.convertTo(stack, CV_32F, 255.0);
    return stack;
}

/**
 * @brief Remap 1st series of data to 2nd series, by linearly interpolating new values at positions of 1st series and value range of 2nd series.
 * 
 * @param refSeries array-like input 1st series (any size)
 * @param targetSeries array-like input 2nd series (any size)
 * @return Container<T1> array-like series with interpolated values in range of targetSeries and refSeries size
 */
template <template <typename> class Container, 
          typename T1>
Container<T1> remapSeries(const Container<T1>& refSeries, const Container<T1>& targetSeries) {
  using namespace std;
  using namespace cv;

  Container<T1> refPositions(refSeries);
  auto[_Y, _indUnique] = unique(targetSeries);
  Container<T1> Y(_Y);
  Container<T1> X = Y;
  refPositions = NormalizeTo_0_1(refPositions);
  X = NormalizeTo_0_1(X);

  auto sortedIndices = Argsort(X);
  X = Reorder(X, sortedIndices);
  Y = Reorder(Y, sortedIndices);
  // TODO: interpolators possibly cant extrapolate and fill with 0 when out of range
  _1D::LinearInterpolator<float> interp1; //LinearDelaunayTriangleInterpolator  ThinPlateSplineInterpolator
  interp1.setData(X,Y);
  Container<T1> results(refSeries.size());
  for(int x=0; x<refPositions.size(); x++) {
    float y = interp1((T1)refPositions[x]);
    results[x] = y;
  }

  return results;
}

