#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/traits.hpp"
#include "opencv2/core/types.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/imgproc.hpp"
#include "opencv2/highgui.hpp"
#include "opencv2/video.hpp"
#include "opencv2/calib3d.hpp"
#include "opencv2/photo.hpp"
#include <cmath>
#include <string>
#include <tuple>
#include <valarray>
#include <vector>
#include <libInterpolate/Interpolate.hpp>
#include "utils.cpp"

#pragma once
#pragma GCC target ("avx")
// #pragma GCC optimize ("unroll-loops")


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
  calibrate_color,
  benchmark
};

struct ProgramParams {
  int VERBOSITY = 0;
  ProgramTask task = ProgramTask::stack; //stack
  // ProgramTask task = ProgramTask::benchmark;
  float radicalChangeRatio = 2.0f;
  int interpolation = cv::INTER_LANCZOS4; //which interpolation algorithm to use (1:Linear, 4:Lanchos) //4 //cv::INTER_LINEAR; cv::INTER_LANCZOS4
  int erosion_size = 1; //expand mask of bad pixels //3
};
int VERBOSITY = 0; //0

struct AlignmentParams {
  int base_index = -1; //index of base reference image //0
  double checkArea = 0.75; //image comparison area //0.7
  double alpha = 0.8; //how many points to keep for alignment //1.0
  int maxIter = 50; //max number of undistortion iterations //30
  bool alignCenter = false; //keep center of images the same //false
  bool warpAlign = true; //apply warp perspective operation to align images //true
  int splitAlignPartsVertical = 1; //how many times to split image (vertically) to align each part independently //4
  int splitAlignPartsHorizontal = 1; //how many times to split image (horizontally) to align each part independently //4
  int warpIter = 40; //max number of align image iterations //0
  int n_points = 16000; //initial number of points to detect and compare between images //1024
  int K = -1;  //number of points clusters for estimating distortion //3
  float ratio = 0.8f; //how many points to keep for undistortion //0.65f,
  bool mirroring = false; //try mirroring best alignment //false
};

struct StackingParams {
  int patternN = 200; //number of sharpness checking regions (total=patternN*patternN) //200
  int patternSize = 7; //size of each sharpness checking region //3
  float minImgCoef = 0.0f; //minimum value to add to each image's coefficients //0.0
  float baseImgCoef = 0.4f; //coefficient value of base image (by default first img is base) //0.5f
  float coef_sharpness = 1.5; //local sharpness weight for total image coeffs //1.0
  float coef_similarity = 1.0; //local similarity to base img weight for total image coeffs //1.0
  double comparison_scale = 0.5; //pixel ratio - decrease resolution for calculating some parameters //1.0
  int blur_size = 5; //adds smoothing to coefficients (increase it to hide switching pixel regions between images)
  double upscale = 1.0; //if value is greater than 1.0 then final image will be upscaled (by upscaling input images and merging them) //1.0
  float discardRatio = 0.2; //how many photos with worst similarity to discard from stacking (increase it when images ale not well aligned)
};

struct ColorParams {
  int histSize = 65; //number or color values per channel //32 //64 
  int num_dominant_colors = 16; //how many colors to use for alignment 3
  int find_colors = 20; //how many colors to search for best match //num_dominant_colors*1.5
  float strength = 1.0f; //how much to change/align the color //1.0f
  float maxChange = 0.15; //limit ratio (original*(1+maxChange)) for max color change //0.1f
};


std::string to_string(ProgramTask task)
{
  std::string result = "-1";
  switch (task)
  {
    case ProgramTask::stack: 
      result = "stack";
      break;
    case ProgramTask::simple_stack: 
      result = "simple_stack";
      break;
    case ProgramTask::align: 
      result = "align";
      break;
    case ProgramTask::calibrate_color: 
      result = "calibrate_color";
      break;
    case ProgramTask::benchmark: 
      result = "benchmark";
      break;
  }

  return result;
}

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

  std::vector<int> indices(vec2.size());
  std::iota(indices.begin(), indices.end(), 0);
  sort(indices.begin(), indices.end(), [&](int i, int j) { return isGreater(vec2[i], vec2[j]); });
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


/**
 * @brief Initialize 2D maps with values in range [0...n].
 * 
 * @param rows 
 * @param cols 
 * @return std::tuple<cv::Mat, cv::Mat> Pair of 2D maps of float32 type. 1st for transposition of first dimension and 2nd for transposition of 2nd dim.
 */
std::tuple<cv::Mat, cv::Mat> initMap2D(int rows, int cols)
{
  using namespace std;
  using namespace cv;

  Mat X = Mat::zeros(rows, cols, CV_32F);
  Mat Y = Mat::zeros(rows, cols, CV_32F);

  #pragma omp parallel for
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

  if((H.rows < 3) || (H.cols < 3)) {
    if(VERBOSITY > 0) {
      cout << "Warning: warpPerspective requires input matrix of size [3 x 3]. Returning unchanged image."<<endl;
    }
    return image.clone();
  }

  try
  {
    // Mat Hinv = H.inv();
    int width = size.width;
    int height = size.height;

    cv::Mat unwarped(image.size(), image.type());
    cv::warpPerspective(image, unwarped, H, cv::Size(width, height), interpolation, cv::BORDER_CONSTANT, 0);
    return unwarped;
  }
  catch (const std::exception&)
  {
    return image.clone();
  }
}

cv::Mat Undistort(const cv::Mat &image, const cv::Mat &mtx, const cv::Mat &dist, int width, int height, cv::InterpolationFlags interpolation=cv::INTER_LINEAR,
                  double alpha = 1.0, bool AlignCenter = true)
{
  using namespace cv;
  cv::Mat unwarped = image;
  cv::Size oryginalCameraResolution(width,height);
  // cv::Size oryginalCameraResolution(mtx.at<float>(0, 2)*2.0, mtx.at<float>(1, 2)*2.0);
  auto newCameraResolution = cv::Size(image.cols, image.rows); // cv::Size(width, height); //cv::Size(image.cols, image.rows); //oryginalCameraResolution;
  //TODO: improve getting newcameramtx
  cv::Mat newcameramtx = mtx;
  Mat identityMat = Mat::eye(3, 3, CV_32F);
  // if(std::equal(mtx.begin<float>(), mtx.end<float>(), identityMat.begin<float>())) {
  //   newcameramtx = mtx;
  // }
  // else {
    newcameramtx = cv::getOptimalNewCameraMatrix(mtx, dist, oryginalCameraResolution, alpha, newCameraResolution);
    // newcameramtx.at<float>(0, 2) = width/2;
    // newcameramtx.at<float>(1, 2) = height/2;
    // newcameramtx.at<float>(0, 0) = (float)width; //(float)std::min(width,height);
    // newcameramtx.at<float>(1, 1) = (float)height; //(float)std::min(width,height);
    // if(VERBOSITY > 1) {
    //   std::cout << "Undistort: Warning: Using nondefault matrix" << std::endl;
    // }
  // }

  if(AlignCenter) {
    //TODO: possible error
    cv::Mat H = cv::Mat::eye(3, 3, CV_32F);
    H.at<float>(0, 2) = (float)-newcameramtx.at<float>(0,2); //(float)(newCameraResolution.width/2) - newcameramtx.at<float>(0,2);
    H.at<float>(1, 2) = (float)-newcameramtx.at<float>(1,2); //(float)(newCameraResolution.height/2) - newcameramtx.at<float>(1,2);
    unwarped = warpPerspective(image, H, newCameraResolution, interpolation);
  }
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

  try
  {
    // cv::Mat unwarped = image.clone();
    cv::Mat unwarped;
    cv::BorderTypes border = cv::BorderTypes::BORDER_CONSTANT;
    cv::remap(image, unwarped, map1, map2, interpolation, border, 0);
    return unwarped;
  }
  catch (const std::exception&)
  {
    return image.clone();
  }
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
  else if(isEqual(str, "benchmark", true)) {
    task = ProgramTask::benchmark;
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
  if(loadeddict.isMember("discardRatio")) {
    stackPars1.discardRatio = loadeddict["discardRatio"].as<float>();
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

void SaveImage(const cv::Mat &image, const std::string &pathToFile="img", const std::string &fileType=".jpg", bool overwrite=false, bool addTimestamp=true, bool RGB2BGR=false) {
  using namespace std;
  using namespace cv;


  // Check if results folders exist
  namespace fs = std::filesystem;
  auto pathToFile2 = ReplaceAll(pathToFile, "//", "/");
  auto words = split(pathToFile2, '/');
  words = selectFirstN(words, (int)words.size()-1);
  if(words.size() > 0) {
    auto pathToDir = concatenate(words, "/");
    pathToDir = "./" + pathToDir;
    pathToDir = ReplaceAll(pathToDir, "././", "./");
    if (!fs::is_directory(pathToDir) || !fs::exists(pathToDir)) { 
      fs::create_directory(pathToDir);
    }
  }
  pathToFile2 = "./" + pathToFile2;
  pathToFile2 = ReplaceAll(pathToFile2, "././", "./");
  string filePath = "";
  if(addTimestamp) {
    time_t timestamp = time(NULL);
    struct tm datetime = *localtime(&timestamp);
    char dateFormatted[16];
    strftime(dateFormatted, 16, "%Y%m%d_%H%M%S", &datetime);
    filePath = pathToFile2+"_"+dateFormatted+"_"+fileType;
  }
  else {
    filePath = pathToFile2 + fileType;
  }
  
  Mat _image = image;

  if( (_image.type() != CV_8U) && 
      (_image.type() != CV_8UC1) && 
      (_image.type() != CV_8UC2) && 
      (_image.type() != CV_8UC3) && 
      (_image.type() != CV_8UC4) ) {
    double maxVal, minVal;
    cv::minMaxLoc(_image, &minVal, &maxVal);
    double ratio = maxVal/255.0;
    if(minVal < 0.0) {
      ratio = (maxVal-minVal)/255.0;
      _image.convertTo(_image, CV_8U, 1.0, minVal);
    }
    _image.convertTo(_image, CV_8U, 1.0/ratio);
  }

  if(RGB2BGR) {
    cvtColor(_image, _image, COLOR_RGB2BGR);
  }

  //dont overwrite existing file
  if(!overwrite) {
    if(!filesystem::exists(filePath)) {
      imwrite(filePath, _image);
    }
    //TODO: suggest new name (with number?) for file with the same name
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

cv::Mat ReduceResolution(const cv::Mat inputImage, int maxRes = 1000, int interpolation = cv::INTER_LINEAR)
{
  cv::Mat image;
  if ((std::max(inputImage.rows, inputImage.cols) > maxRes) && (inputImage.rows > inputImage.cols))
  {
    int newHeight = maxRes;
    int newWidth = int(maxRes * (static_cast<double>(inputImage.cols) / inputImage.rows));
    cv::resize(inputImage, image, cv::Size(newWidth, newHeight), 0, 0, interpolation);
  }
  else if ((std::max(inputImage.rows, inputImage.cols) > maxRes) && (inputImage.rows <= inputImage.cols))
  {
    int newHeight = int(maxRes * (static_cast<double>(inputImage.rows) / inputImage.cols));
    int newWidth = maxRes;
    cv::resize(inputImage, image, cv::Size(newWidth, newHeight), 0, 0, interpolation);
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
    std::valarray<T> distances = varr - mean;
    distances = distances * distances;
    return distances.sum() / distances.size();
}

cv::Mat add(const cv::Mat& _array, const cv::Mat& _array2) {
  cv::Mat result;
  cv::Mat array, array2;
  _array.convertTo(array,  CV_32F);
  _array2.convertTo(array2,  CV_32F);
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

cv::Mat add(const cv::Mat& _array, float value2) {
  cv::Mat result;
  cv::Mat array;
  _array.convertTo(array,  CV_32F);
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

cv::Mat subtract(const cv::Mat& _array, const cv::Mat& _array2) {
  cv::Mat result;
  cv::Mat array, array2;
  _array.convertTo(array,  CV_32F);
  _array2.convertTo(array2,  CV_32F);
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

cv::Mat subtract(const cv::Mat& _array, float value2) {
  cv::Mat result;
  cv::Mat array;
  _array.convertTo(array,  CV_32F);
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

cv::Mat subtract(float value, const cv::Mat& _array) {
  cv::Mat result;
  cv::Mat array;
  _array.convertTo(array,  CV_32F);
  if(array.channels() == 1) {
    cv::subtract(value, array, result);
  }
  else if(array.channels() > 1) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < array.channels(); i++) {
      cv::Mat delta;
      cv::Mat channel;
      extractChannel(array, channel, i);
      cv::subtract(value, channel, delta);
      vec.push_back(delta);
    }
    result = to3dMat(vec);
  }

  return result;
}

cv::Mat multiply(const cv::Mat& array, const cv::Mat& array2) {
  cv::Mat _array,_array2,result;
  array.convertTo(_array,  CV_32F);
  array2.convertTo(_array2, CV_32F);

  if(_array.channels() == _array2.channels()) {
    cv::Mat value;
    cv::multiply(_array, _array2, value);
    result = value;
  }
  else if(_array.channels() > _array2.channels()) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < _array.channels(); i++) {
      cv::Mat value;
      cv::Mat channel;
      extractChannel(_array, channel, i);
      cv::multiply(channel, _array2, value);
      vec.push_back(value);
    }
    result = to3dMat(vec);
  }
  else if(_array.channels() < _array2.channels()) {
    std::vector<cv::Mat> vec;
    for (int i = 0; i < _array2.channels(); i++) {
      cv::Mat value;
      cv::Mat channel;
      extractChannel(_array2, channel, i);
      cv::multiply(_array, channel, value);
      vec.push_back(value);
    }
    result = to3dMat(vec);
  }

  return result;
}

cv::Mat multiply(const cv::Mat& array, float value2) {
  using namespace std;
  using namespace cv;
      
  Mat _array;
  array.convertTo(_array, CV_32F);
  cv::Mat result;
  if(_array.channels() == 1) {
    cv::Mat value = Mat::ones(_array.rows, _array.cols, CV_32F);
    cv::multiply(_array, value, value, (double)value2);
    result = value;
  }
  else if(_array.channels() > 1) {
    std::vector<cv::Mat> vec;
    cv::split(_array, vec);
    for (int i = 0; i < _array.channels(); i++) {
      cv::Mat value = Mat::ones(_array.rows, _array.cols, CV_32F);
      cv::Mat channel = vec[i];
      cv::multiply(channel, value, value, (double)value2);
      vec.push_back(value);
    }
    result = to3dMat(vec);
  }

  result.convertTo(result, array.type());
  return result;
}

cv::Mat divide(const cv::Mat& _array, const cv::Mat& _array2) {
  cv::Mat result;
  cv::Mat array, array2;
  _array.convertTo(array,  CV_32F);
  _array2.convertTo(array2,  CV_32F);
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

cv::Mat divide(float value2, const cv::Mat& _array) {
  using namespace std;
  using namespace cv;

  cv::Mat result;
  cv::Mat array;
  _array.convertTo(array,  CV_32F);
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

/**
 * @brief Round input 3d Mat to nearest integer
 * 
 * @param array 3d cv::Mat of type float
 * @return 3d cv::Mat of type float with numbers rounded to integers
 */
cv::Mat round(const cv::Mat& array) {
  using namespace std;
  using namespace cv;

  cv::Mat result;
  std::vector<cv::Mat> vec;
  for (int i = 0; i < array.channels(); i++) {
    cv::Mat channel;
    extractChannel(array, channel, i);
    for (int r = 0; r < array.rows; r++)
    {
      for (int c = 0; c < array.cols; c++)
      {
        channel.at<float>(r,c) = round(channel.at<float>(r,c));
      }
    }
    vec.push_back(channel);
  }
  result = to3dMat(vec);

  return result;
}

/**
 * @brief Function takes vector of Mats and returns single Mat with summed values (sum along all Mats)
 * 
 * @param series vector of Mats
 * @return single cv::Mat with summed values
 */
cv::Mat sumSeriesTo1(const std::vector<cv::Mat>& series) {
  using namespace std;
  using namespace cv;
  cv::Mat img = cv::Mat::zeros(series[0].rows, series[0].cols, CV_32F);
  for(int c=0; c<series.size(); c++) {
    img = add(img,series[c]);
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

  // #pragma ivdep
  #pragma omp parallel for
  for(int c=0; c<channels.size(); c++) {
    #pragma ivdep
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
        result[i] = (result[i] - minVal) / delta2;
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
    //TODO: implement other axis
    if(axis==2) {
      array[0].convertTo(min2D, CV_32F);
      array[0].convertTo(max2D, CV_32F);
      cv::Mat current2D; //cv::Mat::zeros(array[i].rows, array[i].cols, cv::CV_32F);
      for (int i = 1; i < array.size(); i++) {
        // current2D = array[i];
        array[i].convertTo(current2D, CV_32F);
        cv::min(current2D,min2D,min2D);
        cv::max(current2D,max2D,max2D);        
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
/// @param mask Multichannel mask is treated as 1 big mask (not prosessing each channel separately)
/// @return Mask of input size, float32 type and inverted values
cv::Mat invertMask(const cv::Mat &mask) {
  using namespace std;
  using namespace cv;

  Mat invMask = mask.clone();
  invMask.convertTo(invMask, CV_32F);
  double minVal,maxVal;
  minMaxLoc(invMask, &minVal, &maxVal);
  invMask = subtract(invMask, maxVal);
  invMask = cv::abs(invMask);

  return invMask;
}

/// @brief Compares and thresholds each channel of image to specified color. Default mask value is 1.0f, matching regions are set to 0.0f.
/// @param image input multichannel image
/// @param color optional: input multichannel color (default is 0)
/// @return Mask with size of image (1 channel), float32 type and values in range [0.0f; 1.0f] inclusive
cv::Mat maskFromColor(const cv::Mat &image, std::vector<int> color={0,0,0}) {
  using namespace std;
  using namespace cv;

  Mat _image;
  image.clone().convertTo(_image, CV_8U);
  vector<Mat> channels;
  Mat mask = Mat::zeros(image.rows, image.cols, CV_32F);
  cv::split(image.clone(), channels);
  for (int i = 0; i < channels.size(); i++)
  {
    Mat current = Mat::ones(image.rows, image.cols, CV_32F);
    for (int r = 0; r < channels[i].rows; r++)
    {
      for (int c = 0; c < channels[i].cols; c++)
      {
        if(channels[i].at<uint8_t>(r,c) == (uint8_t)(color[i])) {
          current.at<float>(r,c) = 0.0f;
        }
      }
    }
    mask = add(mask, current);
  }
  cv::threshold(mask, mask, (double)0.0, 1.0, THRESH_BINARY);
  mask = NormalizeTo_0_1(mask);
  return mask;
}

//TODO: fix comparison with detectionRatio < 1.0
/// @brief Divides each channel of image1 by image2 and checks if value is within detectionRatio. Default mask value is 1.0f, matching regions are set to 0.0f.
cv::Mat maskFromChange(const cv::Mat &image1, const cv::Mat &image2, float detectionRatio, float resolutionScale = 1.0f) {
  using namespace std;
  using namespace cv;

  Mat mask, current;
  vector<Mat> channels;
  int w = image1.cols;
  int h = image1.rows;
  float _detectionRatio;
  if(detectionRatio < 1.0f) {
    _detectionRatio = 1.0f/detectionRatio;
  }
  else {
    _detectionRatio = detectionRatio;
  }
  if(resolutionScale) {
    w = (int)round((double)(w) * resolutionScale);
    h = (int)round((double)(h) * resolutionScale);
  }
  mask = Mat::ones(h, w, CV_32F);
  current = Mat::ones(h, w, CV_32F);
  Mat image1_ = Resize(image1, w, h);
  Mat image2_ = Resize(image2, w, h);
  image1_.convertTo(image1_, CV_32F);
  image2_.convertTo(image2_, CV_32F);
  Mat result = divide(image1_, image2_);
  cv::split(result, channels);
  for (int i = 0; i < channels.size(); i++)
  {
    current = Mat::ones(h, w, CV_32F);
    #pragma omp parallel for
    for(int r = 0; r < result.rows; r++)
    {
      for(int c = 0; c < result.cols; c++) {
        if(channels[i].at<float>(r,c) <= (1.0f/_detectionRatio)) {
          current.at<float>(r,c) = 0.0f;
        }
      }        
    }      
    #pragma omp parallel for
    for(int r = 0; r < result.rows; r++)
    {
      for(int c = 0; c < result.cols; c++) {
        if(channels[i].at<float>(r,c) >= _detectionRatio) {
          current.at<float>(r,c) = 0.0f;
        }
      }        
    }      
    mask = multiply(mask, current);
  }

  mask = Resize(mask, image1.cols, image1.rows, INTER_NEAREST);
  mask = NormalizeTo_0_1(mask);
  return mask;
}

/**
 * @brief Select pixels from input image with mask equal to 1 and set other pixels to 0.
 * 
 * @param image1 input image
 * @param mask mask with values in range [0; 1]
 * @return cv::Mat masked image
 */
cv::Mat selectMasked(const cv::Mat& image1, const cv::Mat& mask) {
  using namespace std;
  using namespace cv;

  Mat result = multiply(image1, mask);
  result.convertTo(result, image1.type());

  return result;
}

/**
 * @brief Select pixels from first image with mask equal to 1 and pixels from second image with mask equal to 0.
 * 
 * @param image1 first image
 * @param image2 second image
 * @param mask mask with values in range [0; 1]
 * @return cv::Mat masked image
 */
cv::Mat selectMasked(const cv::Mat& image1, const cv::Mat& image2, const cv::Mat& mask) {
  using namespace std;
  using namespace cv;

  Mat _image1 = image1.clone();
  Mat _image2 = image2.clone();
  _image1.convertTo(_image1, CV_32F);
  _image2.convertTo(_image2, CV_32F);
  Mat result = multiply(_image1, mask);
  result = add(result, multiply(_image2, invertMask(mask)));
  result.convertTo(result, image1.type());

  return result;
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
  int xCut = (int)((1.0 - area2) * 0.5 * (double)imageRef.cols);
  int yCut = (int)((1.0 - area2) * 0.5 * (double)imageRef.rows);
  std::vector<int> topLeftCorner = {0 + yCut, 0 + xCut};
  std::vector<int> topRightCorner = {0 + yCut, imageRef.cols - xCut};
  std::vector<int> bottomLeftCorner = {imageRef.rows - yCut, 0 + xCut};
  std::vector<int> bottomRightCorner = {imageRef.rows - yCut, imageRef.cols - xCut};
  cv::Mat image1 = CutImgToCorners(imageRef, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
  cv::Mat image2 = CutImgToCorners(imageTest2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
  double mse=0.0;
  for (int i = 1; i < image1.channels(); i++) {
    Mat errorSE, channelImg1, channelImg2; 
    extractChannel(image1, channelImg1, i);
    extractChannel(image2, channelImg2, i);
    channelImg1.convertTo(channelImg1, CV_32F);
    channelImg2.convertTo(channelImg2, CV_32F);
    absdiff(channelImg1, channelImg2, errorSE);
    Scalar s = sum(errorSE); // sum  per channel
    mse += (double)(s.val[0]);  
  }
  mse = mse / (double)(image1.rows * image1.cols * image1.channels());
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

double CompareImgQuick(const cv::Mat &imageRef, const cv::Mat &imageTest, double area = 1.0)
{

  using namespace std;
  using namespace cv;

  Mat image1 = imageRef;
  Mat image2 = imageTest;
  if (image1.channels() > 1)
  {
    cv::cvtColor(image1, image1, cv::COLOR_BGR2GRAY);
  }
  if (image2.channels() > 1)
  {
    cv::cvtColor(image2, image2, cv::COLOR_BGR2GRAY);
  }

  double area2 = area;
  if(area > 1.0) {
    area2 = area2 / 100.0;
  }
  if(area < 0.0) {
    area2 = 1.0;
  }
  int xCut = (int)((1.0 - area2) * 0.5 * imageRef.cols);
  int yCut = (int)((1.0 - area2) * 0.5 * imageRef.rows);
  std::vector<int> topLeftCorner = {0 + yCut, 0 + xCut};
  std::vector<int> topRightCorner = {0 + yCut, imageRef.cols - xCut};
  std::vector<int> bottomLeftCorner = {imageRef.rows - yCut, 0 + xCut};
  std::vector<int> bottomRightCorner = {imageRef.rows - yCut, imageRef.cols - xCut};
  image1 = CutImgToCorners(image1, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
  image2 = CutImgToCorners(image2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);


  Mat errorSE;
  absdiff(image1, image2, errorSE);
  errorSE.convertTo(errorSE, CV_32F);
  Scalar s = sum(errorSE); // sum  per channel
  double se = (s.val[0]) / (double)(image1.rows * image1.cols);
  return se;
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

//TODO: Split detection and matching of points
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
    cvtColor(img1, grayImg1, COLOR_BGR2GRAY);
  }
  else {
    grayImg1 = img1;
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
    if(VERBOSITY > 1 ) {
      cout<<"DetectFeatures: warning - empty descriptors of img1"<<endl;
    }
    return make_pair(cv::Mat{}, cv::Mat{});
  }
  if(descriptors2.empty() || descriptors1.rows < 8) { 
    if(VERBOSITY > 1 ) {
      cout<<"DetectFeatures: error - empty descriptors of img2"<<endl;
    }
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
  #pragma ivdep
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
    if(VERBOSITY > 1) {
      cout << "DetectFeatures: WARNING: Failed to detect points" << endl;
    }
    return make_pair(matchpoints1_, matchpoints2_);
  }
  matchpoints1.col(0).copyTo(matchpoints1_.col(0));
  matchpoints1.col(1).copyTo(matchpoints1_.col(1));
  matchpoints2.col(0).copyTo(matchpoints2_.col(0));
  matchpoints2.col(1).copyTo(matchpoints2_.col(1));

  // Draw matches
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

  // Draw matches
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

/// @brief Detect and describe points between 2 images using specified method
/// @return pairs of points and their descriptors for img1 and img2
std::tuple<std::vector<cv::KeyPoint>, cv::Mat, std::vector<cv::KeyPoint>, cv::Mat> DetectImageFeatures(const cv::Mat &img1, const cv::Mat &img2, int nfeatures=32000, 
                                                                                                        const cv::Mat &mask=cv::Mat(), const Detector &method=Detector::orb,
                                                                                                        float scaleFactor=1.5f, int nlevels=5) {
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
    detector = AKAZE::create(cv::AKAZE::DESCRIPTOR_MLDB, 0, 1);
  }
  else {
    detector = ORB::create(nfeatures);
  }

  Mat grayImg1;
  if(img1.channels() > 1) {
    cvtColor(img1, grayImg1, COLOR_BGR2GRAY);
  }
  else {
    grayImg1 = img1;
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
    if(VERBOSITY > 1 ) {
      cout<<"DetectFeatures: warning - empty descriptors of img1"<<endl;
    }
    return make_tuple(vector<KeyPoint>{}, cv::Mat{}, vector<KeyPoint>{}, cv::Mat{});
  }
  if(descriptors2.empty() || descriptors1.rows < 8) { 
    if(VERBOSITY > 1 ) {
      cout<<"DetectFeatures: error - empty descriptors of img2"<<endl;
    }
    return make_tuple(vector<KeyPoint>{}, cv::Mat{}, vector<KeyPoint>{}, cv::Mat{});
  }
  descriptors1.convertTo(descriptors1, CV_32F);
  descriptors2.convertTo(descriptors2, CV_32F);
  
  return make_tuple(keypoints1, descriptors1, keypoints2, descriptors2);
}

/// @brief Match points between 2 images
/// @return pairs of matched points in img1 and img2
std::tuple<cv::Mat, cv::Mat> MatchImageFeatures(const std::vector<cv::KeyPoint>& keypoints1, const cv::Mat &descriptors1, const std::vector<cv::KeyPoint>& keypoints2, const cv::Mat &descriptors2, float Lowes=0.77f, int K=2) {
  using namespace std;
  using namespace cv;

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
    return make_tuple(cv::Mat{}, cv::Mat{});
  }

  vector<DMatch> matches;
  for (const auto &pair : possibleMAtches) {
      if (pair[0].distance < Lowes * pair[1].distance)
          matches.push_back(pair[0]);
  }

  Mat matchpoints1 = Mat::zeros(matches.size(), 2, CV_32F);
  Mat matchpoints2 = Mat::zeros(matches.size(), 2, CV_32F);
  #pragma ivdep
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
    if(VERBOSITY > 1) {
      cout << "DetectFeatures: WARNING: Failed to detect points" << endl;
    }
    return make_tuple(matchpoints1_, matchpoints2_);
  }
  matchpoints1.col(0).copyTo(matchpoints1_.col(0));
  matchpoints1.col(1).copyTo(matchpoints1_.col(1));
  matchpoints2.col(0).copyTo(matchpoints2_.col(0));
  matchpoints2.col(1).copyTo(matchpoints2_.col(1));

  return make_tuple(matchpoints1_, matchpoints2_);
}

cv::Mat _warpImage(const cv::Mat &inputImage, const cv::Size &orygSize, const cv::Mat &M, 
                          cv::Mat* outMask=nullptr, 
                          int borderMode=cv::BORDER_CONSTANT, 
                          cv::InterpolationFlags interp=cv::InterpolationFlags::INTER_LANCZOS4, 
                          bool affine=false,
                          bool exactBorders=false) {
  using namespace std;
  using namespace cv;
  
  try
  {
    Mat _inputImage = inputImage.clone();
    Mat H = M.clone();
    int flags = (int)interp;

    if (affine) {
      H = H.rowRange(0, 2);
    }

    int h = orygSize.height;
    int w = orygSize.width;
    cv::Mat unwarped;
    if (borderMode == cv::BORDER_CONSTANT) {
      if (affine) {
        cv::warpAffine(_inputImage, unwarped, H, cv::Size(w, h), flags, cv::BORDER_CONSTANT, 0);
      } else {
        if(exactBorders) {
          Mat map1,map2;
          Mat R;
          Mat dist = Mat::zeros(14,1, CV_32F); //Mat::ones(14,1, CV_32F);
          initUndistortRectifyMap(H, dist, R, H, cv::Size(w, h), CV_32FC1, map1, map2);
          map1 = round(map1);
          map2 = round(map2);
          unwarped = Undistort(_inputImage, map1, map2, interp);
        }
        else {
          cv::warpPerspective(_inputImage, unwarped, H, cv::Size(w, h), flags, cv::BORDER_CONSTANT, 0);
        }
      }
    } else {
      if (affine) {
          cv::warpAffine(_inputImage, unwarped, H, cv::Size(w, h), flags, borderMode);
      } else {
        if(exactBorders) {
          Mat map1,map2;
          Mat R;
          Mat dist = Mat::ones(5,1, CV_32F);
          initUndistortRectifyMap(H, dist, R, H, unwarped.size(), CV_32FC1, map1, map2);
          map1 = round(map1);
          map2 = round(map2);
          unwarped = Undistort(_inputImage, map1, map2, interp);
        }
        else {
          cv::warpPerspective(_inputImage, unwarped, H, cv::Size(w, h), flags, borderMode);
        }
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
      cv::threshold(mask,	mask, 0.0, 1.0, THRESH_BINARY);
      (*outMask) = mask;
    }

    return unwarped;
  }
  catch (const std::exception&)
  {
    if(VERBOSITY > 0) {
      cout << "Warning: warpPerspective requires input matrix of size [3 x 3]. Returning unchanged image."<<endl;
    }
    return inputImage.clone();
  }
}

cv::Mat AlignImageToImage(const cv::Mat &refImage, const cv::Mat &inputImage, 
                          cv::InterpolationFlags interp=cv::InterpolationFlags::INTER_LANCZOS4, 
                          int nfeatures=4000, 
                          float scaleFactor=1.5f, 
                          int nlevels=5, 
                          float Lowes=0.75f, 
                          int K=3, 
                          float ransacReprojThreshold=10.0f, 
                          cv::Mat* inMask=nullptr, 
                          int borderMode=cv::BORDER_CONSTANT,
                          int eccIter=0, 
                          bool affine=false,                          
                          cv::Mat* M=nullptr, 
                          bool exactBorder=false) {
  using namespace std;
  using namespace cv;
  
  Mat _inputImage = inputImage.clone();
  auto [pts0, pts1] = DetectFeatures(_inputImage, refImage, nfeatures,cv::Mat(),Detector::orb,scaleFactor,nlevels,Lowes,K);
  // auto [pts0, pts1] = DetectFeatures2(_inputImage, refImage,100,0.005,10.0,true);
  if((pts0.rows < 8) || (pts1.rows < 8)) {
    if(VERBOSITY > 1) {
      cout << "AlignImageToImage: WARNING: Failed to align image. Returning unchanged input image." <<endl;
    }
    return _inputImage.clone();
  }

  auto H = cv::findHomography(pts0, pts1, cv::RANSAC, ransacReprojThreshold);
  // auto H = cv::findHomography(pts0, pts1, cv::RHO, ransacReprojThreshold);

  cv::InterpolationFlags flags = interp;
  if (eccIter > 1) {
    Mat orygGray;
    Mat inputGray;
    if(refImage.channels() > 1) {
      cv::cvtColor(refImage, orygGray, cv::COLOR_BGR2GRAY);
    }
    else {
      orygGray = refImage.clone();
    }
    if(_inputImage.channels() > 1) {
      cv::cvtColor(_inputImage, inputGray, cv::COLOR_BGR2GRAY);
    }
    else {
      inputGray = _inputImage.clone();
    }
    flags = interp;
    H = cv::Mat(H).inv();
    H.convertTo(H, CV_32F);
    cv::TermCriteria criteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, eccIter, 1e-4);
    try
    {
      if (inMask != nullptr) {
        cv::findTransformECC(orygGray, inputGray, H, cv::MOTION_HOMOGRAPHY, criteria, *inMask);
      } else {
        cv::findTransformECC(orygGray, inputGray, H, cv::MOTION_HOMOGRAPHY, criteria);
      }
    }
    catch(const std::exception& e)
    {
      // std::cerr << e.what() << '\n';
      if(VERBOSITY > 1) {
        cout<<"AlignImageToImage: skipping ecc alignment" << endl;
      }
    }
    H = cv::Mat(H).inv();
    H.convertTo(H, CV_32F);
  } else {
    flags = interp;
  }

  Mat unwarped = _warpImage(_inputImage, refImage.size(), H,
                          inMask, borderMode, flags, affine, exactBorder);

  if(M != nullptr) {
    *M = H;
  }

  //try also with default settings
  Mat unwarped2 = Mat::zeros(unwarped.size(), unwarped.type());
  try
  {
    auto [ptsx2, ptsy2] = DetectFeatures(_inputImage, refImage);
    auto H2 = cv::findHomography(ptsx2, ptsy2);
    unwarped2 = _warpImage(_inputImage, refImage.size(), H2);
  }
  catch (const std::exception&)
  {
  }

  //check for improvement
  // double mse1 = CompareImgQuick(refImage, _inputImage, 0.8);
  // double mse2 = CompareImgQuick(refImage, unwarped, 0.8);
  double mse1 = CompareImg(refImage, _inputImage, 0.5);
  double mse2 = CompareImg(refImage, unwarped, 0.5);
  double mse3 = CompareImg(refImage, unwarped2, 0.5);
  vector<double> msePerImage {mse1, mse2, mse3};
  vector<Mat> unwarpedCandidates {_inputImage, unwarped, unwarped2};
  auto sortedIndices = Argsort(msePerImage);
  int best_ind = sortedIndices[0];
  unwarped = unwarpedCandidates[best_ind].clone();
  if(best_ind == 0) {
    if(VERBOSITY > 1) {
      cout << "AlignImageToImage: Warning: Failed to align image. Returning unchanged input image." <<endl;
    }
  }

  return unwarped;
}

std::tuple<cv::Mat, cv::Mat> AlignImageToImageRegions(const cv::Mat &refImage, const cv::Mat &inputImage, cv::Size2i num_parts=cv::Size2i(2,2), 
                          AlignmentParams aparams=AlignmentParams(), cv::Mat* inMask=nullptr, cv::InterpolationFlags interpolation=cv::INTER_LANCZOS4)
{
  using namespace std;
  using namespace cv;
  
  int borderMode=cv::BORDER_CONSTANT;
  auto imageSize = refImage.size();
  int width = (int)(imageSize.width / num_parts.width);
  int height = (int)(imageSize.height / num_parts.height);
  vector<vector<Rect>> rois;
  vector<vector<Mat>> images;
  vector<vector<Mat>> images2;
  vector<vector<Mat>> imagesAligned(num_parts.width, vector<Mat>(num_parts.height));
  vector<vector<Mat>> masksAligned(num_parts.width, vector<Mat>(num_parts.height));
  rois.resize(num_parts.width);
  images.resize(num_parts.width);
  images2.resize(num_parts.width);

  for (int i = 0; i < num_parts.width; i++)
  {
    for (int j = 0; j < num_parts.height; j++)
    {
      cv::Rect roi = cv::Rect((i*width),(j*height), width,height);
      Mat truncated;
      Mat truncated2;
      refImage(roi).clone().copyTo(truncated);
      inputImage(roi).clone().copyTo(truncated2);
      rois[i].push_back(roi);
      images[i].push_back(truncated);
      images2[i].push_back(truncated2);
    }
  }
  
  #pragma omp parallel for
  for (int i = 0; i < num_parts.width; i++)
  {
    for (int j = 0; j < num_parts.height; j++)
    {
      Mat aligned;
      try
      {
        // aligned = AlignImageToImage(images[i][j], images2[i][j], 
        //                             INTER_LANCZOS4, 
        //                             4000, 1.1f, 16, 0.7, 
        //                             3, 5.0f,
        //                             nullptr, 
        //                             BORDER_CONSTANT, 
        //                             30,
        //                             false, 
        //                             nullptr, 
        //                             true );
        aligned = AlignImageToImage(images[i][j], images2[i][j], 
                                    interpolation, 
                                    aparams.n_points, 1.2f, 16, aparams.ratio, 
                                    3, 5.0f,
                                    nullptr, 
                                    BORDER_CONSTANT, 
                                    aparams.warpIter,
                                    false);
      }
      catch(const std::exception& e)
      {
        try
        {
          aligned = AlignImageToImage(images[i][j], images2[i][j], INTER_LANCZOS4);
        }
        catch(const std::exception& e)
        {
          aligned = images2[i][j];
          // std::cerr << e.what() << '\n';
        }
      }
      Mat mask = Mat::zeros(aligned.size(), CV_32F);
      mask = add(maskFromColor(aligned), invertMask(maskFromColor(images[i][j])));
      cv::threshold(mask,	mask, 0.0, 1.0, THRESH_BINARY);
      mask.convertTo(mask, CV_32F);
      // aligned = selectMasked(aligned, images[i][j], mask);
      imagesAligned[i][j] = aligned;
      masksAligned[i][j] = mask;
    }
  }
  
  Mat result = Mat::zeros(imageSize, inputImage.type());
  Mat mask = Mat::ones(imageSize, CV_32F);
  for (int i = 0; i < num_parts.width; i++)
  {
    for (int j = 0; j < num_parts.height; j++)
    {
      imagesAligned[i][j].copyTo(result(rois[i][j]));
      masksAligned[i][j].copyTo(mask(rois[i][j]));
    }
  }

  return make_tuple(result, mask);
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
    valarray<float> zero((float)0.0f, cols);
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
  //TODO: add propher method for selecting randomly K points
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
        continue;
      }
      else {
        s.insert(vec[i]);
        uniquePoints.push_back(vec[i]);
        uniqueIndices.push_back(i);
      }
    }    

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
  result[pos1] = target[pos2];
  result[pos2] = target[pos1];

  return result;
}

// Function to swap columns
// @returns cloned cv::Mat with swaped col1 and col2
cv::Mat swapCol(const cv::Mat &target, int col1, int col2) {
  cv::Mat result = target.clone();
  target.col(col1).copyTo(result.col(col2));
  target.col(col2).copyTo(result.col(col1));
  return result;
}

/**
 * @brief Create mask with borders set to value and center to 1.
 * 
 * @param height total height 
 * @param width total width
 * @param topLeftMask start point of mask
 * @param botRightMask end point of mask
 * @param value value for masked region (default=0)
 * @return cv::Mat mask of type float with borders set to @value 
 */
cv::Mat borderMask(int height, int width, cv::Point2i startMask, cv::Point2i endMask, float value=0.0f)
{
  using namespace std;
  using namespace cv;

  Mat mask = Mat::ones(height, width, CV_32F);
  rectangle(mask, startMask, endMask, Scalar_<float>{value}, -1);

  return mask;
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

cv::Mat featherMask(const cv::Mat &mask, int range=1, float minVal=0.0f, float maxVal=1.0f) {
  using namespace std;
  using namespace cv;

  Mat mask2 = mask.clone();
  Mat feathered_mask = mask.clone();
  Mat randomized = mask.clone();
  Mat element = getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3));
  erode(mask2, mask2, element, Point2i(-1,-1), range);
  feathered_mask = subtract(feathered_mask, mask2);    
  cv::randu(randomized, minVal, maxVal);
  cv::blur(randomized, randomized, cv::Size(3, 3));
  feathered_mask = multiply(feathered_mask, randomized);
  return add(mask2, feathered_mask);
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
  map1 = patchNaNs(map1, (double)0.0,(double)0.0,(double)0.0);
  map2 = patchNaNs(map2, (double)0.0,(double)0.0,(double)0.0);
  map1.convertTo(map1, CV_32F);
  map2.convertTo(map2, CV_32F);
    // medianBlur(map1, map1, 3);
    // medianBlur(map2, map2, 3);
  auto stop = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);

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
    cv::cvtColor(refImg, img2, cv::COLOR_BGR2GRAY);
  }
  if(distortedImg.channels() > 1) {
    cv::cvtColor(distortedImg, dist2, cv::COLOR_BGR2GRAY);
  }
  //reduce resolution for faster processing
  if(refImg.rows >= 4000) {
    img2 = Resize(img2, refImg.cols/4, refImg.rows/4);
    dist2 = Resize(dist2, refImg.cols/4, refImg.rows/4);
  }
  else if(refImg.rows >= 1024) {
    img2 = Resize(img2, refImg.cols/2, refImg.rows/2);
    dist2 = Resize(dist2, refImg.cols/2, refImg.rows/2);
  }
  maskImg = maskFromColor(img2);
  mse_ref = CompareImgQuick(img2, dist2, checkArea);
  mseList.push_back(mse_ref);
  resolutions.push_back({dist2.cols, dist2.rows});
  if(VERBOSITY > 1) {
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
  cv::Mat aligned = dist2.clone();
  int widthROI = (int)(checkArea * (double)imageSize.width);
  int heightROI = (int)(checkArea * (double)imageSize.height);
  cv::Rect *roi = new cv::Rect((imageSize.width-widthROI)/2,(imageSize.height-heightROI)/2, widthROI,heightROI);
  Rect imageCenter = *roi;
  Mat maskCenter(imageSize, CV_8UC1, Scalar::all(0));
  maskCenter(imageCenter).setTo(Scalar::all(255));
  if(warpAlign) {
    // TODO: add alignimage mask
    //TODO: add alignimage for map1,map2
    Mat aligned1, aligned2;
    // if(!useCenterMask) {
    aligned1 = AlignImageToImage(img2,dist2, 
                          INTER_LINEAR, 
                          n_points, 1.2f, 16, ratio, 3, 
                          10.0f, nullptr, BORDER_CONSTANT, warpIter);
    float mse_dist1 = CompareImgQuick(img2, aligned1, checkArea);
    if(mse_dist1 < mse_ref) {
      aligned = aligned1;
      mse_dist = mse_dist1;
      useCenterMask = false;
    }
    // }
    // else {
     aligned2 = AlignImageToImage(img2,dist2, 
                          INTER_LINEAR, 
                          n_points, 1.2f, 16, ratio, 3, 
                          10.0f, &maskCenter, BORDER_CONSTANT, warpIter);
    float mse_dist2 = CompareImgQuick(img2, aligned2, checkArea);
    if((mse_dist2 < mse_ref) && (mse_dist2 < mse_dist1)) {
      aligned = aligned2;
      mse_dist = mse_dist2;
      useCenterMask = true;
    }
    // }
    // mse_dist = CompareImgQuick(img2, aligned, checkArea);
    // if(mse_dist > mse_ref) {
    //   aligned = dist2;
    // }
    mseList.push_back(mse_dist);
    resolutions.push_back({aligned.cols, aligned.rows});
  }
  // else {
  //   aligned = dist2;
  // }

  // int n_points = 4096; //8192; 1024; 2048;
  unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
  std::default_random_engine generator(seed);
  std::uniform_real_distribution<float> distribution((float)0.25f,(float)2.0f);
  std::uniform_real_distribution<float> distribution2((float)0.85f,(float)1.1f);

  if (maxIter > 0)
  {
    dist2 = aligned.clone();
    maskDist = maskFromColor(dist2);  
    // Mat maskCombined = multiply(maskImg, maskDist);
    Mat maskCombined = add(maskDist, maskImg);
    cv::threshold(maskCombined,	maskCombined, 0.0, 1.0, THRESH_BINARY);
    maskCombined.convertTo(maskCombined, CV_32F);
    // maskCombined = invertMask(maskCombined);
    maskCombined.convertTo(maskCombined, CV_8U, 255.0);
    mse_dist = CompareImgQuick(img2, dist2, checkArea);
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
      // objectPoints.resize(listSize+gatherIter);
      // imagePoints.resize(listSize+gatherIter);
      objectPoints.resize(listSize+1);
      imagePoints.resize(listSize+1);
      // vector<Mat> points1(gatherIter);
      // vector<Mat> points2(gatherIter);
      vector<vector<KeyPoint>> points1(gatherIter);
      vector<vector<KeyPoint>> points2(gatherIter);
      vector<Mat> _descriptors1(gatherIter);
      vector<Mat> _descriptors2(gatherIter);
      #pragma omp parallel for
      for (int j = 0; j < gatherIter; j++)
      {
        //Find undistortion for aligned images
        int rand_features =  (float)n_points * distribution(generator);
        // float pointsRatio = (float)ratio * distribution2(generator);
        // if(VERBOSITY > 1) {
        //   cout << "calibrateCamera: rand_features=" << rand_features << endl;
        // }      
        //detect points for undistortion
        auto[keypoints1,descriptors1,keypoints2,descriptors2] = DetectImageFeatures(img2, dist2, 
                                                                                                                                rand_features, 
                                                                                                                                Mat(), 
                                                                                                                                Detector::orb, 
                                                                                                                                1.3f, 
                                                                                                                                8);
        if((descriptors1.rows < 8) || (descriptors2.rows < 8)) {
          if(VERBOSITY > 1) {
            cout << "RelativeUndistort: skipping aligment with incorrect points" << endl;
          }
          continue;
        }
        points1[j] = keypoints1;
        points2[j] = keypoints2;
        _descriptors1[j] = descriptors1;
        _descriptors2[j] = descriptors2;
      }
      Mat descriptors1 = _descriptors1[0];
      Mat descriptors2 = _descriptors2[0];
      for (int j = 1; j < gatherIter; j++) {
        if((_descriptors1[j].rows > 8) && (_descriptors2[j].rows > 8)) {
          vconcat(descriptors1, _descriptors1[j], descriptors1);
          vconcat(descriptors2, _descriptors2[j], descriptors2);
        }
      }
      auto keypoints1 = flatten(points1);
      auto keypoints2 = flatten(points2);
      float pointsRatio = (float)ratio * distribution2(generator);
      auto[matchedPoints1_,matchedPoints2_] = MatchImageFeatures(keypoints1, descriptors1, keypoints2, descriptors2, pointsRatio, 3);                                                                                                               
      objectPoints[listSize] = toVecPoint3f(matchedPoints1_);
      imagePoints[listSize] = toVecPoint2f(matchedPoints2_);

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
      // if(VERBOSITY > 1) {
      //   cout << "RelativeUndistort: combined_points.size() = " << combined_points.size() << endl; 
      // }

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
      // undistorted = Undistort(dist2, newMap1, newMap2, interp);
      undistorted = Undistort(dist2, newMap1, newMap2);
      if(!useCenterMask) {
        undistorted = AlignImageToImage(img2,undistorted, INTER_LINEAR);
      }
      else {
        undistorted = AlignImageToImage(img2, undistorted, 
                          INTER_LINEAR, 
                          4000, 1.5f, 5, 0.75f, 3, 
                          10.0f, &maskCenter);
      }
      double mse = CompareImgQuick(img2, undistorted, checkArea);

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
        double mse_topLeft = CompareImgQuick(CutImgToCorners(img2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner), topLeft, checkArea);
        topLeftCorner = {0, w/2}; //Point2i(0,0); //{0, w/2};
        topRightCorner = {0, w-1}; //Point2i(w-1,0); //{0, w-1};
        bottomLeftCorner = {h/2, w/2}; //Point2i(0,h/2); //{h/2, w/2};
        bottomRightCorner = {h/2, w-1}; //Point2i(w-1,h/2); //{h/2, w-1};
        cv::Rect topRight_roi = CornersToRect(topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        topRight = CutImgToCorners(undistorted, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        topRight_newMap1 = CutImgToCorners(newMap1, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        topRight_newMap2 = CutImgToCorners(newMap2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        double mse_topRight = CompareImgQuick(CutImgToCorners(img2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner), topRight, checkArea);
        topLeftCorner = {h/2, 0};
        topRightCorner = {h/2, w/2};
        bottomLeftCorner = {h-1, 0};
        bottomRightCorner = {h-1, w/2};
        cv::Rect bottomLeft_roi = CornersToRect(topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        bottomLeft = CutImgToCorners(undistorted, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        bottomLeft_newMap1 = CutImgToCorners(newMap1, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        bottomLeft_newMap2 = CutImgToCorners(newMap2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        double mse_bottomLeft = CompareImgQuick(CutImgToCorners(img2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner), bottomLeft, checkArea);
        topLeftCorner = {h/2, w/2};
        topRightCorner = {h/2, w-1};
        bottomLeftCorner = {h-1, w/2};
        bottomRightCorner = {h-1, w-1};
        cv::Rect bottomRight_roi = CornersToRect(topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        bottomRight = CutImgToCorners(undistorted, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        bottomRight_newMap1 = CutImgToCorners(newMap1, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        bottomRight_newMap2 = CutImgToCorners(newMap2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner);
        double mse_bottomRight = CompareImgQuick(CutImgToCorners(img2, topLeftCorner, topRightCorner, bottomLeftCorner, bottomRightCorner), bottomRight, checkArea);
        
        vector<double> mse_partial = {mse_topLeft, mse_topRight, mse_bottomLeft, mse_bottomRight};
        vector<Rect> partial_roi = {topLeft_roi, topRight_roi, bottomLeft_roi, bottomRight_roi};
        vector<Mat> partial_newMap1 = {topLeft_newMap1, topRight_newMap1, bottomLeft_newMap1, bottomRight_newMap1};
        vector<Mat> partial_newMap2 = {topLeft_newMap2, topRight_newMap2, bottomLeft_newMap2, bottomRight_newMap2};
        int ind = min_element(mse_partial.begin(), mse_partial.end()) - mse_partial.begin();
        newMap1 = mirror4way(partial_newMap1, ind);
        newMap2 = mirror4way(partial_newMap2, ind);
        newMap1 = MatchResolution(newMap1, imageSize, interp);
        newMap2 = MatchResolution(newMap2, imageSize, interp);
        // Mat undistorted2 = Undistort(dist2, newMap1, newMap2, interp);
        Mat undistorted2 = Undistort(dist2, newMap1, newMap2);
        double mse2 = CompareImgQuick(img2, undistorted2, checkArea);
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
          //show clusters
          Mat figure = img2.clone();
            for(auto p : center_points){
              cv::circle(figure, p, 15, {100, 255, 100}, 5);
            }
            for(auto p2 : center_points2){
              cv::circle(figure, p2, 10, {150, 150, 255}, 3);
            }
          show(figure,"Detected features - Cluster centers", 5000);
          SaveImage(figure, "./Detected features - Cluster centers"); //test
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
        // // auto flags_clustering = cv::KMEANS_PP_CENTERS; //Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007]
        // // double compactness = kmeans(matchedPoints1, 64, bestLabels, termCriteria, 200, flags_clustering, centers);
        // // double compactness2 = kmeans(matchedPoints2, 64, bestLabels2, termCriteria, 200, flags_clustering, centers2);
        // centers.convertTo(centers, CV_32F);        
        // centers2.convertTo(centers2, CV_32F);
        // auto[newMap21,newMap22] = calibrateCamera(centers, centers2, imageSize, resolutionRatio, false);
        vector< vector<Point3f> > objectPoints2(0, vector<Point3f>());
        vector< vector<Point2f> > imagePoints2(0, vector<Point2f>()); // 2d points in new image
        objectPoints2.push_back(toVecPoint3f(centers));
        imagePoints2.push_back(toVecPoint2f(centers2));
        auto[newMap21,newMap22, mtx2,distortion2] = calibrateCameraOCV(objectPoints2, imagePoints2, imageSize, 
                                                                              checkArea, flags, mtx, distortion);
        newMap1 = newMap21;
        newMap2 = newMap22;
        Mat undistorted2 = Undistort(undistorted, 
                            subtract(add(map1,newMap1), map1_ref), 
                            subtract(add(map2,newMap2), map2_ref));
        double mse3 = CompareImgQuick(img2, undistorted2, checkArea);
        // Mat test1 = Undistort(dist2, map1, map2, interp);
        Mat test1 = Undistort(dist2, map1, map2);
        double test_mse1 = CompareImgQuick(img2, test1, checkArea);
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
            SaveImage(figure, "./Detected features - Cluster centers - method2"); //test

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
    SaveImage(map1, "./calibrateCamera"+string("_map1"));
    SaveImage(map2, "./calibrateCamera"+string("_map2"));
    SaveToCSV(mseList, "./RelativeUndistort_mseList.csv");
  }

  // Apply undistortion for final image
  // aligned = distortedImg.clone();
  maskCenter = Resize(maskCenter, refImg.cols, refImg.rows, INTER_NEAREST);
  if(warpAlign) {
    // TODO: add alignimage mask
    if(!useCenterMask) {
      aligned = AlignImageToImage(refImg, distortedImg, 
                                  interp, 
                          n_points, 1.2f, 16, ratio, 3, 
                          10.0f, nullptr, BORDER_CONSTANT, warpIter);
    }
    else {
      aligned = AlignImageToImage(refImg, distortedImg, 
                                  interp, 
                          n_points, 1.2f, 16, ratio, 3, 
                          10.0f, &maskCenter, BORDER_CONSTANT, warpIter);
    }
  }
  undistorted = Undistort(aligned, map1, map2, interp);
  // maskDist = maskFromColor(undistorted);
  maskDist = add(maskFromColor(undistorted), invertMask(maskFromColor(refImg)));
  cv::threshold(maskDist,	maskDist, 0.0, 1.0, THRESH_BINARY);
  maskDist.convertTo(maskDist, CV_32F);

  maskDist.convertTo(maskDist, CV_8U, 255.0);
  undistorted = AlignImageToImage(refImg,undistorted, 
                                  interp, 
                      n_points, 1.1f, 5, ratio, 2, 
                      5.0f, &maskDist, BORDER_CONSTANT, warpIter,
                      false);
  cv::Mat mask;
  cv::Mat distortedImgGray;
  cv::cvtColor(aligned, distortedImgGray, cv::COLOR_BGR2GRAY);
  // TODO: fix masking for new undistortion algorithm

  bool badUndistortion = false;
  cv::Mat undistoredGray;
  cv::cvtColor(undistorted, undistoredGray, cv::COLOR_BGR2GRAY);
  mask = add(maskFromColor(undistorted), invertMask(maskFromColor(refImg)));
  cv::threshold(mask,	mask, 0.0, 1.0, THRESH_BINARY);
  mask.convertTo(mask, CV_32F);

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

  double mse_final = CompareImgQuick(refImg, undistorted, checkArea);

  if(VERBOSITY > 1) {
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

  auto points = toValarray(image);
  float meanSharpness = variance(points);

  auto points2 = toVec(points);
  int half_size = points2.size();
  sort(begin(points2), end(points2));
  auto part1 = toValarray(selectFirstN(points2, half_size));
  auto part2_ = selectLastN(points2, half_size);
  reverse(part2_.begin(), part2_.end());
  auto part2 = toValarray(part2_);
  auto distances = abs(part2-part1);
  float total_diff(distances.sum() / (float)half_size);
  return meanSharpness * total_diff;
}

cv::Mat sharpnessOfRegions(const cv::Mat& image, float patternSize, int patternN) {
  cv::Mat image2;
  if (image.channels() > 1) { 
      cv::cvtColor(image, image2, cv::COLOR_BGR2GRAY); // cv::cvtColor(image, image2, cv::COLOR_BGR2HSV);
  } else {
      image2 = image.clone();
  }
  if(image2.type() != CV_32F) {
    image2.convertTo(image2, CV_32F);
  }

  int height, width;
  if (patternSize < 1.0f) {
      height = std::max(int(image2.rows * patternSize), 3);
      width = std::max(int(image2.cols * patternSize), 3);
  } else {
      height = std::max(int(patternSize), 3);
      width = std::max(int(patternSize), 3);
  }

  std::vector<int> rows(patternN);
  std::vector<int> cols(patternN);
  #pragma ivdep
  // #pragma GCC ivdep
  // #pragma Clang loop vectorize(enable)
  for (int i = 0; i < patternN; ++i) {
      rows[i] = std::round(height + i * (image2.rows - 2 * height) / (patternN - 1));
      cols[i] = std::round(width + i * (image2.cols - 2 * width) / (patternN - 1));
  }
  std::vector<std::vector<float>> sharpnessPerRegion(rows.size(), std::vector<float>(cols.size()));
  #pragma ivdep
  // #pragma GCC ivdep
  // #pragma Clang loop vectorize(enable)
  // #pragma omp parallel for
  for (int r = 0; r < rows.size(); r++)
  {
    std::vector<float> sharpnessPerCol(cols.size());
    #pragma ivdep
    // #pragma GCC ivdep
    // #pragma Clang loop vectorize(enable)
    for (int c = 0; c < cols.size(); c++)
    {
        cv::Mat region = image2(cv::Rect(cols[c], rows[r], width, height));
        sharpnessPerCol[c] = sharpness(region);
    }
    sharpnessPerRegion[r] = sharpnessPerCol;
  }

  cv::Mat sharpnessPerRegion_lowRes = toMat(sharpnessPerRegion);
  cv::Mat sharpnessPerRegion_highRes;
  cv::resize(sharpnessPerRegion_lowRes, sharpnessPerRegion_highRes, cv::Size(image2.cols,image2.rows));

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

/**
 * @brief Find max value per pixel in multi-channel image
 * 
 * @param image multi-channel image
 * @return cv::Mat single-channel image with max values
 */
cv::Mat maxFromChannels(const cv::Mat& image) {
  using namespace std;
  using namespace cv;

  Mat max2D;
  extractChannel(image, max2D, 0);
  cv::Mat current2D;
  for (int i = 1; i < image.channels(); i++) {
    extractChannel(image, current2D, i);
    cv::max(current2D,max2D,max2D);        
  }

  return max2D.clone();
}

/// @brief Combines multiple images into single image with maximized sharpness.
/// @param images vector of images to be stacked
/// @param base_id optional which image is treated as base (by default first img is base)
/// @param imagesMasks optional vector of image masks
/// @return single combined image with 3 channels
cv::Mat stackImages( const std::vector<cv::Mat>& images, int base_index=0, const std::vector<cv::Mat>& imagesMasks=std::vector<cv::Mat>(), 
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
    double minTest, maxTest;

    std::vector<cv::Mat> images2(images.size());
    #pragma omp parallel for
    for (int i = 0; i < images.size(); i++) {
        images2[i] = images[i].clone();
      // images2[i] = limitChange(images[base_id], images[i], radicalChangeRatio);
    }
    std::vector<Mat> masks2 = imagesMasks;
    if(masks2.size()<images.size()) {
      for (int i = 0; i < (images.size() - imagesMasks.size()); i++) {
        Mat baseValue = Mat::ones(images[base_index].rows,images[base_index].cols, CV_32F);
        masks2.push_back(baseValue);
      }
    }

    //estimate sharpness of input images
    std::vector<Mat> sharpnessPerImage(images2.size()); //Mat sharpnessPerImage;
    std::vector<float> averageSharpnessPerImage(images2.size());
    Mat sharpnessRefImg;
        // auto start = std::chrono::high_resolution_clock::now();
    //TODO: try using unsharp mask with erode/dialte to fill holes
    #pragma omp parallel for
    for (int i = 0; i < images2.size(); i++) {
        if(i==base_index) {
            sharpnessRefImg = sharpnessOfRegions(images2[base_index], patternSize, patternN);
            sharpnessRefImg.convertTo(sharpnessRefImg, CV_32F);
            sharpnessPerImage[i] = sharpnessRefImg;
            averageSharpnessPerImage[i] = cv::mean(sharpnessRefImg)[0];
            continue;
        }
        cv::Mat img = images2[i];
        // cv::Mat img = multiply(images2[i], masks2[i]);
        auto sharpnesPerRegion = sharpnessOfRegions(img, patternSize, patternN);
        sharpnesPerRegion.convertTo(sharpnesPerRegion, CV_32F);
        sharpnesPerRegion = multiply(sharpnesPerRegion, masks2[i]);
        sharpnessPerImage[i] = sharpnesPerRegion;
        averageSharpnessPerImage[i] = cv::mean(sharpnesPerRegion)[0];
    }
      // if(VERBOSITY > 1) {
      //   auto stop = std::chrono::high_resolution_clock::now();
      //   auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
      //   std::cout << "StackImages: estimating sharpness of images "<< " | elapsed time = "<< MicroToSeconds(duration.count()) << " [s]" << std::endl;  
      // }
    Mat sharpnessAll = to3dMat(sharpnessPerImage);
    sharpnessPerImage.clear();

    //estimate diffrences of input images
    std::vector<cv::Mat> diffrencePerImage(images2.size());
    std::vector<float> msePerImage(images2.size());
    cv::Mat baseImgGray;
    cv::cvtColor(images2[base_index].clone(), baseImgGray, cv::COLOR_BGR2GRAY);
    #pragma omp parallel for
    for (int i = 0; i < images2.size(); i++) {
      if(i==base_index) {
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
        baseImgGray2 = Resize(baseImgGray, scaledW, scaledH, INTER_AREA); //area is good for downscaling
        ImgGray2 = Resize(ImgGray, scaledW, scaledH, INTER_AREA); //area is good for downscaling
      }
      cv::Mat diffrences = cv::abs(baseImgGray2 - ImgGray2);
      diffrences = Resize(diffrences, baseImgGray.size().width, baseImgGray.size().height, interpolation);
      diffrences.convertTo(diffrences, CV_32F);
      // diffrences = multiply(diffrences, masks2[i]);
      diffrencePerImage[i] = diffrences;
      float mse = CompareImg(baseImgGray2, ImgGray2, 0.8);
      msePerImage[i] = mse;
    }

    //discard worst input images
    {
      vector<Mat> _imagesForProcessing;
      vector<Mat> _masks;
      vector<float> _mseList;
      auto indices = Argsort(msePerImage);
      for (int i = 0; i < images2.size(); i++) {
        if( (i == base_index) || 
            (i < 2) ||
            (i < (int)( (float)(images2.size()) - sparams.discardRatio * (float)(images2.size()) )) ) {
          _imagesForProcessing.push_back(images2[indices[i]]);
          _masks.push_back(masks2[indices[i]]);
          _mseList.push_back(msePerImage[indices[i]]);
        }
      }
      images2 = _imagesForProcessing;
      masks2 = _masks;
      msePerImage = _mseList;
    }
   
    // normalize each point for all photos
    Mat diffrenceAll = to3dMat(diffrencePerImage);
    Mat coeffSum2d = sumSeriesTo1(diffrencePerImage);
    diffrencePerImage.clear();
    diffrenceAll = divide(diffrenceAll, coeffSum2d);
    vector<Mat> temp = toVecMat(sharpnessAll);    
    coeffSum2d = sumSeriesTo1(temp);
    sharpnessAll = divide(sharpnessAll, coeffSum2d);
    sharpnessAll = patchNaNs(sharpnessAll, (float)1.0f / images2.size(), (float)0.0f, (float)1.0f);
    // diffrenceAll = patchNaNs(diffrenceAll, (float)1.0f, (float)0.0f, (float)1.0f); //cv::patchNaNs(diffrenceAll, (float)1.0f);
    valarray<float> temp2 = toValarray(NormalizeTo_0_1(msePerImage)) + 1.0f;
    msePerImage = toVec(temp2);
    float minVal = 0.01;
    std::vector<cv::Mat> similarityPerImage(images2.size());
    #pragma omp parallel for
    for (int i = 0; i<images2.size(); i++) {
      if((i==base_index) && (baseImgCoef > 0.0)) {
        Mat baseValue = Mat(diffrenceAll.rows, diffrenceAll.cols, CV_32F, (float)baseImgCoef);
        similarityPerImage[base_index] = baseValue;
        continue;
    }
        Mat similarityOfImg;
        extractChannel(diffrenceAll, similarityOfImg, i);
        similarityOfImg = patchNaNs(similarityOfImg, (float)1.0f/similarityPerImage.size());
        similarityOfImg = divide(1.0f, add(similarityOfImg, minVal));
        similarityOfImg.convertTo(similarityOfImg, CV_32F);
        similarityOfImg = multiply(similarityOfImg, similarityOfImg); //enhance diffrences between good and bad image regions
        similarityOfImg = multiply(similarityOfImg, masks2[i]);
        similarityPerImage[i] = similarityOfImg;
    }
    diffrenceAll.release();

    //fix wrong values
    #pragma omp parallel for
    for (int i=0;i<similarityPerImage.size();i++) {
      if(i == base_index) {
        similarityPerImage[i] = patchNaNs(similarityPerImage[i], (float)1.0f, (float)1.0f, (float)1.0f);
      }
      else {
        similarityPerImage[i] = patchNaNs(similarityPerImage[i], (float)1.0f/similarityPerImage.size());
      }
    }

    //sum of coeffs per pixel should be 1.0
    coeffSum2d = sumSeriesTo1(similarityPerImage);
    #pragma omp parallel for
    for (int i=0;i<similarityPerImage.size();i++) {
      similarityPerImage[i] = divide(similarityPerImage[i], coeffSum2d);
    }

    // coeffSum2d = sumSeriesTo1(similarityPerImage);
    // similarityPerImage = toVecMat(divide(to3dMat(similarityPerImage), coeffSum2d));
    // similarityPerImage = NormalizeTo_0_1(similarityPerImage, 2);
    similarityPerImage = toVecMat(patchNaNs(to3dMat(similarityPerImage), (float)max(similarityPerImage), (float)0.0f, (float)1.0f));

    //apply sharpness coefficient
    std::vector<cv::Mat> imageCoeffs(images2.size());
    #pragma omp parallel for
    for (int i = 0; i < images2.size(); i++) {
        Mat sharpnessImg;
        extractChannel(sharpnessAll, sharpnessImg, i);
        imageCoeffs[i] = multiply(sharpnessImg, coef_sharpness); //coef_sharpness * sharpnessImg;
        if(i == base_index) {
          imageCoeffs[i] = add(imageCoeffs[i], (float)minImgCoef);
        }
    }
    // imageCoeffs[0] = add(similarityPerImage[0], coef_similarity); //coef_similarity * cv::Mat::ones(imageCoeffs[i].rows, 1, CV_32F);
    sharpnessAll.release();

    //apply similarity coefficient
    #pragma omp parallel for
    for (int i = 0; i < images2.size(); i++) {        
        imageCoeffs[i] = add(imageCoeffs[i], multiply(similarityPerImage[i], coef_similarity));
    }
    similarityPerImage.clear();
    coeffSum2d = sumSeriesTo1(imageCoeffs);

    //apply mask to image coeffs
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      if(i != base_index) {
        imageCoeffs[i] = multiply(imageCoeffs[i], masks2[i]);
      }
    }

    //sum of coeffs per pixel should be 1.0
    coeffSum2d = sumSeriesTo1(imageCoeffs);
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      imageCoeffs[i] = divide(imageCoeffs[i], coeffSum2d);
    }

    // set worst coeffs to 0  //TODO:find min pixels in multichannel image
    // auto replaced = Replace(to3dMat(imageCoeffs), (float)0.0f, ImgReplaceMode::minimum);
    // imageCoeffs = toVecMat(replaced);

    //fix wrong values
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      if(i == base_index) {
        imageCoeffs[i] = patchNaNs(imageCoeffs[i], (float)1.0f, (float)1.0f, (float)1.0f);
      }
      else {
        imageCoeffs[i] = patchNaNs(imageCoeffs[i], (float)0.0f, (float)0.0f, (float)0.0f);
      }
    }

    //apply masks
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      imageCoeffs[i] = multiply(imageCoeffs[i], masks2[i]);      
    }

    //increase diffrences between coeffs
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      imageCoeffs[i] = multiply(imageCoeffs[i], imageCoeffs[i]); //increase diffrences between images
    }

    //make smooth transition between pixels
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      // auto coeff = imageCoeffs[i];
      // cv::medianBlur(coeff, coeff, 3);
      cv::blur(imageCoeffs[i], imageCoeffs[i], cv::Size(blur_size, blur_size));
    //   imageCoeffs[i] = coeff;
    }

    //apply masks to coeffs
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      imageCoeffs[i] = multiply(imageCoeffs[i], masks2[i]);      
    }

    //sum of coeffs per pixel should be 1.0
    coeffSum2d = sumSeriesTo1(imageCoeffs);
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      imageCoeffs[i] = divide(imageCoeffs[i], coeffSum2d);
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

    //use only pixels that represent best 50% of weights
    Mat max2D = maxFromChannels( to3dMat(imageCoeffs) );
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      Mat topMask = maskFromChange( imageCoeffs[i], max2D, 4.0f, 0.5f);
      imageCoeffs[i] = multiply(imageCoeffs[i], topMask);      
    }

    //sum of coeffs per pixel should be 1.0
    coeffSum2d = sumSeriesTo1(imageCoeffs);
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      imageCoeffs[i] = divide(imageCoeffs[i],coeffSum2d); //sum of coeffs for given pixel should be 1.0
    }

    //fix wrong values
    #pragma omp parallel for
    for (int i=0;i<imageCoeffs.size();i++) {
      if(i == base_index) {
        imageCoeffs[i] = patchNaNs(imageCoeffs[i], (float)1.0f, (float)1.0f, (float)1.0f);
      }
      else {
        imageCoeffs[i] = patchNaNs(imageCoeffs[i], (float)0.0f, (float)0.0f, (float)0.0f);
      }
    }

    Mat stack = Mat::zeros(images2[0].rows, images2[0].cols, CV_32F);
    for (int i = 0; i < images2.size(); i++) {
      cv::Mat img2;
      images2[i].convertTo(img2, CV_32F);
      img2 = multiply(img2, imageCoeffs[i]);
      stack = add(stack,img2);
    }
    cv::minMaxLoc(stack, &minTest, &maxTest);
    imageCoeffs.clear();
    stack.convertTo(stack, CV_32F, 255.0/maxTest);

    cv::Mat sharpnesPerRegion = sharpnessOfRegions(stack, patternSize, patternN);
    cv::Mat sharpnessDiff = sharpnesPerRegion - sharpnessRefImg;
    sharpnessDiff.convertTo(sharpnessDiff, CV_32F);
    double betterSharp = cv::sum(sharpnessDiff >= 0.0)[0] / (stack.rows * stack.cols);
    sharpnesPerRegion.release();
        // cv::minMaxLoc(sharpnessDiff, &minTest, &maxTest);
        // if(VERBOSITY > 1) {
        //   cout << "StackImages: stacked - sharpnessDiff: minVal=" << minTest << " ; maxVal=" << maxTest << endl;
        // }
    
    //Restore oryginal pixels for regions with reduced sharpnes
    cv::Mat badIndices;
    cv::Mat goodIndices;
    // double res = cv::threshold(sharpnessDiff, goodIndices, 0.0, 1.0, THRESH_BINARY); //values 0 or greater are good
    cv::inRange(sharpnessDiff, 0.0f, numeric_limits<float>::infinity(), goodIndices);
    // goodIndices.convertTo(goodIndices, CV_32F, 1.0/255.0);
    goodIndices = NormalizeTo_0_1(goodIndices);
    // // Mat radicalChange = maskFromChange(images2[base_id], stack, radicalChangeRatio);
    // Mat radicalChange = maskFromChange(images2[base_id], stack, radicalChangeRatio, sparams.comparison_scale);
    // goodIndices = multiply(goodIndices, radicalChange);
    //   // Mat mask = goodIndices.clone();
    //   // Mat randomized = goodIndices.clone();
      goodIndices.convertTo(goodIndices, CV_8U, 255.0);
      Mat element = getStructuringElement(MorphShapes::MORPH_RECT,
                                Size(3, 3));
      // erode(goodIndices, goodIndices, element, Point2i(-1,-1), pparams.erosion_size);
      // goodIndices.convertTo(goodIndices, CV_32F, 1.0/255.0);
    goodIndices = NormalizeTo_0_1(goodIndices);

      // mask = subtract(mask, goodIndices);    
      // cv::randu(randomized, 0.0f, 1.0f);
      // mask = multiply(mask, randomized);
      // goodIndices = add(goodIndices, mask);
      // cv::blur(goodIndices, goodIndices, cv::Size(3, 3));
      // goodIndices = NormalizeTo_0_1(goodIndices);
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
    baseImgRestoring = images[base_index].clone();
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
    stackChannels[0] = add(multiply(stackChannels[0], goodIndices), multiply(baseImgChannels[0], badIndices));
    stackChannels[1] = add(multiply(stackChannels[1], goodIndices), multiply(baseImgChannels[1], badIndices));
    stackChannels[2] = add(multiply(stackChannels[2], goodIndices), multiply(baseImgChannels[2], badIndices));
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

int benchmark_processing() {
  using namespace std;
  using namespace cv;

  cout << "Benchmarking CherryPK Pixel Stacker with default settings" << endl;
  auto start_benchmark = std::chrono::high_resolution_clock::now();

  int result = -1;
  int N = 8; //number of images //8
  int lines = 20; //number of lines in test image //20

  ProgramParams programParams1;
  AlignmentParams alignmentParams1;
  StackingParams stackingParams1;
  ColorParams colorParams1;
    // alignmentParams1.maxIter=0;
    // alignmentParams1.warpIter=0;
    // alignmentParams1.splitAlignPartsHorizontal=0;
    // alignmentParams1.splitAlignPartsVertical=0;
    // stackingParams1.coef_sharpness = 3.0f;
    // stackingParams1.coef_similarity = 1.0f;
    // stackingParams1.comparison_scale = 0.70f;
    // stackingParams1.blur_size = 1;
    // programParams1.radicalChangeRatio = 1.4f;
    // programParams1.erosion_size = 1;
  auto interpolation = (cv::InterpolationFlags)programParams1.interpolation; //cv::INTER_LANCZOS4; //cv::INTER_LINEAR; cv::INTER_LANCZOS4
  float radicalChangeRatio = programParams1.radicalChangeRatio;
  int erosion_size = programParams1.erosion_size; //cut borders of mask
  //Alignment Params
  // int base_index = alignmentParams1.base_index; //index of base reference image
  int base_index = 0; //index of base reference image
  double checkArea = alignmentParams1.checkArea;
  double alpha = alignmentParams1.alpha;
  int maxIter = alignmentParams1.maxIter;
  bool alignCenter = alignmentParams1.alignCenter;
  bool warpAlign = alignmentParams1.warpAlign;
  int warpIter = alignmentParams1.warpIter;
  int K = alignmentParams1.K;
  int n_points = alignmentParams1.n_points;
  float ratio = alignmentParams1.ratio; //how many points to keep for alignment
  bool mirroring = alignmentParams1.mirroring; //try mirroring best alignment
  //Stacking Params
  int patternN = stackingParams1.patternN;
  int patternSize = stackingParams1.patternSize;
  float minImgCoef = stackingParams1.minImgCoef;
  float baseImgCoef = stackingParams1.baseImgCoef;
  float coef_sharpness = stackingParams1.coef_sharpness;
  float coef_similarity = stackingParams1.coef_similarity;
  double comparison_scale = stackingParams1.comparison_scale;
  //Color Params
  int num_dominant_colors = colorParams1.num_dominant_colors; //how many colors to use for alignment
  int histSize = colorParams1.histSize;                
  float colorStrength = colorParams1.strength; //how much to change/align the color
  float maxChange = colorParams1.maxChange; //limit ratio (original*(1+maxChange)) for max color change
  int find_colors = colorParams1.find_colors; //how many colors to search for best match


  //create image
  Mat refImage = Mat::ones(1080,1920,CV_8UC3);
  refImage.convertTo(refImage, CV_8UC3, 0.0, 81.0);
  int w = refImage.cols;
  int h = refImage.rows;
  {
    Point2i bot_left = Point2i{0, h/4};
    Point2i top_right = Point2i{w/4, 0};
    Point2i center = Point2i{w/2, h/2};
    rectangle(refImage, bot_left, top_right, Scalar_<int>{0, 0, 0}, -1);
    bot_left = Point2i{w - w/4, h/4};
    top_right = Point2i{w, 0};
    rectangle(refImage, bot_left, top_right, Scalar_<int>{0, 255, 128}, -1);
    bot_left = Point2i{0, h - h/4};
    top_right = Point2i{w/4, h};
    rectangle(refImage, bot_left, top_right, Scalar_<int>{0, 255, 128}, -1);
    bot_left = Point2i{w - w/4, h - h/4};
    top_right = Point2i{w, h};
    rectangle(refImage, bot_left, top_right, Scalar_<int>{0, 0, 0}, -1);
    bot_left = Point2i{ w/4, h/4};
    top_right = Point2i{3*w/4, 3*h/4};
    rectangle(refImage, bot_left, top_right, Scalar_<int>{255, 255, 255}, -1);
    circle(refImage, center, (w+h)/8, Scalar_<int>{128, 255, 128}, 5);
  }
  for (int i = 0; i < lines; i++)
  {
    Point2i startHorizontal = Point2i{0, (h-1)/lines * i};
    Point2i endHorizontal = Point2i{w-1, (h-1)/lines * i};
    Point2i startVertical = Point2i{(w-1)/lines * i, 0};
    Point2i endVertical = Point2i{(w-1)/lines * i, h-1};
    line(refImage, startHorizontal, endHorizontal, Scalar_<int>{255, 64, 64}, 3);
    line(refImage, startVertical, endVertical, Scalar_<int>{64, 64, 255}, 4);
  }
  string my_txt = "* Benchmarking CherryPK Pixel Stacker *";
  auto font = FONT_HERSHEY_SCRIPT_COMPLEX;
  auto org = Point2i{50, 100};
  auto font_scale = 3;
  auto txt_color = Scalar_<int>{255, 128, 0}; //BGR
  auto thickness = 3;
  putText(refImage, my_txt, org, font, font_scale, txt_color, thickness);
  my_txt = "perfect image is sharp in every region";
  org.y += 100;
  putText(refImage, my_txt, org, font, 2, txt_color, 2);
  my_txt = "all shapes should be clearly defined without ghosting artifacts";
  org.y += 100;
  putText(refImage, my_txt, org, font, 2, txt_color, 2);
  my_txt = "benchmark is making multiple copies of this image and...";
  org.y += 100;
  putText(refImage, my_txt, org, font, 2, txt_color, 2);
  my_txt = "...distorting and blurring them randomly";
  org.y += 100;
  putText(refImage, my_txt, org, font, 2, txt_color, 2);
  my_txt = "images are aligned and stacked to recreate this benchmark image";
  org.y += 100;
  putText(refImage, my_txt, org, font, 2, txt_color, 2);
  if(VERBOSITY > 0) {
    SaveImage(refImage, "./output/testRefImage", ".jpg");
  }


  //create series of images by applying various effects on reference image
  auto start_dataPrep = std::chrono::high_resolution_clock::now();
  vector<Mat> images(N);
  valarray<double> mseList_unprocessed(N);
  // #pragma omp parallel for
  // #pragma ivdep
  // #pragma omp simd
  for (int i = 0; i < N; i++)
  {
    Mat mask = Mat::zeros(h,w,CV_32F);
    Point2i top_left = Point2i{(int)((float)(w) / (float)(N)) * i, 0};
    Point2i bot_right = Point2i{(int)((float)(w) / (float)(N)) * (i+1) -1, h-1 };
    rectangle(mask, top_left, bot_right, 1, -1);
    // mask.convertTo(mask, CV_32F, 1.0/255.0);
    mask.convertTo(mask, CV_32F);
    mask = invertMask(mask);
    Mat image = refImage.clone();
    image.convertTo(image, CV_32F);
    cv::blur(image, image, cv::Size(5, 5));
    image.convertTo(image, CV_8U);
    image = selectMasked(image, refImage, mask);
    if(i == base_index) {
      images[i] = image;
    } 
    else {
      unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
      std::default_random_engine generator(seed);
      std::uniform_real_distribution<float> distributionPix((float)(-min(h,w)) * 0.1f,(float)(min(h,w)) * 0.1f);
      std::uniform_real_distribution<float> distributionRelative((float)-0.1f,(float)0.1f);
      Mat centerMatrix = Mat::eye(3, 3, CV_32F);
      centerMatrix.at<float>(0, 0) = (float)1.0f+distributionRelative(generator);
      centerMatrix.at<float>(1, 1) = (float)1.0f+distributionRelative(generator);
      centerMatrix.at<float>(0, 2) = (float)distributionPix(generator);
      centerMatrix.at<float>(1, 2) = (float)distributionPix(generator);
      image = warpPerspective(image, centerMatrix, Size(w,h));
        // SaveImage(image, "testWarpPerspective", ".jpg", true, false);
      Mat distortion = Mat::zeros(14, 1, CV_32F);
      distortion.at<float>(0, 0) = (float)-6.22;
      distortion.at<float>(1, 0) = (float)-38.14;
      distortion.at<float>(2, 0) = (float)0;
      distortion.at<float>(3, 0) = (float)0;
      distortion.at<float>(4, 0) = (float)254.86;
      distortion.at<float>(5, 0) = (float)-6.36;
      distortion.at<float>(6, 0) = (float)-36.30;
      distortion.at<float>(7, 0) = (float)248.92;
      // Mat test = Undistort(image, Mat::eye(3, 3, CV_32F), distortion, w, h, INTER_LANCZOS4, 1.0, false);
      //   SaveImage(test, "testNoAlignCenter", ".jpg", true, false);
      image = Undistort(image, Mat::eye(3, 3, CV_32F), distortion, w, h, INTER_LANCZOS4, 1.0, true);
        // SaveImage(image, "testForProcessing"+to_string(i), ".jpg", true, false);
        mseList_unprocessed[i] = CompareImg(images[0], image, 0.7);
      images[i] = image;
    }
  }
  auto stop_dataPrep = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop_dataPrep - start_dataPrep);
  std::cout << "BENCHMARK: dataset preparation "<< " | elapsed time: "<< MicroToSeconds(duration.count()) << " [s]" << std::endl;


  Mat baseImg = images[base_index].clone();
  vector<Mat> imagesForProcessing(images.size());
  vector<Mat> masks(N);
  valarray<double> mseList(N);
  //init masks
  Mat baseImgMask = Mat::ones(baseImg.rows, baseImg.cols, CV_32F);
  for (int i = 0; i < N; i++) {
    imagesForProcessing[i] = images[i];
    masks[i] = Mat::ones(imagesForProcessing[i].rows, imagesForProcessing[i].cols, CV_32F);
  }


  auto start = std::chrono::high_resolution_clock::now();
  // Preprocessing images for stacking (undistort)
  if(VERBOSITY > 0) {
    cout << "RelativeUndistort: Finding and interpolating parameters to align/undistort image" << endl;
  }
  #pragma omp parallel for
  for (int i = 0; i < images.size(); i++)
  {
    if(i != base_index) {
      auto [autoUndistorted, mask] = relativeUndistort(baseImg, imagesForProcessing[i], w, h, programParams1, alignmentParams1);
      //mask bad/unknown pixels
      if(erosion_size > 0) {
        Mat element = getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3));
        erode(mask, mask, element, Point2i(-1,-1), max(0, erosion_size) );
      }    
      imagesForProcessing[i] = autoUndistorted;
      masks[i] = mask;
      mseList[i] = CompareImg(multiply(baseImg, mask), multiply(autoUndistorted, mask), 0.7);  
    }
  }
  auto stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "BENCHMARK: undistorion"<< " | elapsed time: "<< MicroToSeconds(duration.count()) << " [s]" << std::endl;


  //upscale images
  int newWidth, newHeight;
  newWidth = (int)(stackingParams1.upscale * (double)baseImg.cols);
  newHeight = (int)(stackingParams1.upscale * (double)baseImg.rows);
  if(stackingParams1.upscale > 1.0) {
    for (int i = 0; i < imagesForProcessing.size(); i++) {
      imagesForProcessing[i] = Resize(imagesForProcessing[i], newWidth, newHeight, interpolation);
      masks[i] = Resize(masks[i], newWidth, newHeight, INTER_NEAREST);
    }
  }
  //realign after upscaling
  Mat baseImg_resized;
  bool realigned2 = false;
  baseImg_resized = Resize(baseImg, newWidth, newHeight, interpolation);
  #pragma omp parallel for
  for (int i = 0; i < imagesForProcessing.size(); i++) {
    if((stackingParams1.upscale > 1.0) && (i != base_index)) {
      Mat autoAligned = AlignImageToImage(baseImg_resized, imagesForProcessing[i]);
      Mat mask = add(maskFromColor(autoAligned), invertMask(maskFromColor(baseImg_resized)));
      cv::threshold(mask,	mask, 0.0, 1.0, THRESH_BINARY);
      mask.convertTo(mask, CV_32F);
      //mask bad/unknown pixels
      if(erosion_size > 0) {
        Mat element = getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3));
        erode(mask, mask, element, Point2i(-1,-1), max(0, erosion_size) );
      }
      imagesForProcessing[i] = autoAligned;
      masks[i] = mask;
      mseList[i] = CompareImg(multiply(baseImg_resized, mask), multiply(autoAligned, mask), 0.7);
      realigned2 = true;
    }
  }

  bool realigned = false;
  #pragma omp parallel for
  for (int i = 0; i < imagesForProcessing.size(); i++) {
    if(i != base_index) {
      if((alignmentParams1.splitAlignPartsHorizontal > 0) && (alignmentParams1.splitAlignPartsVertical > 0) && (i != base_index)) {
        //do alignment with user's settings
        {
          auto [autoAligned, mask] = AlignImageToImageRegions(baseImg_resized, imagesForProcessing[i], 
                                                      Size2i(alignmentParams1.splitAlignPartsHorizontal, alignmentParams1.splitAlignPartsVertical), 
                                                      alignmentParams1, nullptr, interpolation);
          //mask bad/unknown pixels
          if(erosion_size > 0) {
            Mat element = getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3));
            erode(mask, mask, element, Point2i(-1,-1), max(0, erosion_size) );
          }
          imagesForProcessing[i] = autoAligned;
          masks[i] = mask;
          mseList[i] = CompareImg(multiply(baseImg_resized, mask), multiply(autoAligned, mask), 0.7);
          realigned = true;
        }
        //try alignment with default settings and use best image
        {
          auto [autoAligned, mask] = AlignImageToImageRegions(baseImg_resized, imagesForProcessing[i]);
          //mask bad/unknown pixels
          if(erosion_size > 0) {
            Mat element = getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3));
            erode(mask, mask, element, Point2i(-1,-1), max(0, erosion_size) );
          }
          double mse = CompareImg(multiply(baseImg_resized, mask), multiply(autoAligned, mask), 0.7);
          if(mse < mseList[i]) {
            mseList[i] = mse;
            imagesForProcessing[i] = autoAligned;
            masks[i] = mask;
            realigned = true;
          }
        }
        //try alignment of unprocessed input image and use best image
        {
          auto [autoAligned, mask] = AlignImageToImageRegions(baseImg_resized, images[i].clone(), 
                                                      Size2i(alignmentParams1.splitAlignPartsHorizontal, alignmentParams1.splitAlignPartsVertical), 
                                                      alignmentParams1, nullptr, interpolation);
          //mask bad/unknown pixels
          if(erosion_size > 0) {
            Mat element = getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3));
            erode(mask, mask, element, Point2i(-1,-1), max(0, erosion_size) );
          }
          double mse = CompareImg(multiply(baseImg_resized, mask), multiply(autoAligned, mask), 0.7);
          if(mse < mseList[i]) {
            mseList[i] = mse;
            imagesForProcessing[i] = autoAligned;
            masks[i] = mask;
            realigned = true;
          }
        }
      }
    }
  }

  //update masks
  #pragma omp parallel for
  for (int i = 0; i < imagesForProcessing.size(); i++) {
    if(i != base_index) {
      Mat mask = Mat::ones(imagesForProcessing[i].size(), CV_32F);
          // SaveImage(mask, "testMask_before"+to_string(i), ".jpg", true, false);
      mask = NormalizeTo_0_1(mask);
      mask.convertTo(mask, CV_32F);
      //mask bad/unknown pixels
      Mat mask1 = add(maskFromColor(imagesForProcessing[i]), invertMask(maskFromColor(imagesForProcessing[base_index])));
      cv::threshold(mask1,	mask1, 0.0, 1.0, THRESH_BINARY);
      mask = multiply(mask, mask1);
      Mat element = getStructuringElement(MorphShapes::MORPH_RECT, Size(3, 3));
      if(erosion_size > 0) {
        erode(mask, mask, element, Point2i(-1,-1), max(0, erosion_size) );
      }
      Mat mask2 = maskFromChange( imagesForProcessing[base_index], 
                                  imagesForProcessing[i], 
                                  radicalChangeRatio, 
                                  comparison_scale);
      mask2 = featherMask(mask2, 2, 1.0f/3.0f, 1.0f-1.0f/3.0f);
      mask = multiply(mask, mask2);
      mask = NormalizeTo_0_1(mask);
      masks[i] = mask;
      mseList[i] = CompareImg(multiply(imagesForProcessing[base_index], mask), multiply(imagesForProcessing[i], mask), 0.7);
      if(VERBOSITY > 0) {
        SaveImage(mask, "./masks/testMask"+to_string(i), ".jpg");
      }
    }
  }

  //match colors for stacking (basic)
  //match brightness of each channel
  #pragma omp parallel for
  for (int i = 0; i < imagesForProcessing.size(); i++)
  {
    if(i != base_index) {
      vector<float> brightnessRef(3);
      {
        vector<Mat> channels;
        cv::split(imagesForProcessing[base_index], channels);
        for (int c = 0; c < channels.size(); c++)
        {
          brightnessRef[c] = (float)cv::sum(multiply(channels[c], masks[i]))[0];
        }
      }
      vector<float> brightness(3);
      vector<Mat> channels;
      cv::split(imagesForProcessing[i], channels);
      for (int c = 0; c < channels.size(); c++)
      {
        brightness[c] = (float)cv::sum(multiply(channels[c], masks[i]))[0];
        Mat correctedChannel = multiply(channels[c], brightnessRef[c]/brightness[c]);
        channels[c] = correctedChannel;
      }
      Mat corrected = to3dMat(channels);
      corrected.convertTo(corrected, CV_8U);
      imagesForProcessing[i] = corrected;
      Mat mask = masks[i];
      double mse = CompareImg(multiply(imagesForProcessing[base_index], mask), multiply(imagesForProcessing[i], mask), 0.7);
      mseList[i] = mse;
    }
  }

  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "BENCHMARK: alignment"<< " | elapsed time: "<< MicroToSeconds(duration.count()) << " [s]" << std::endl;
  if(VERBOSITY > 0) {
    for (int i = 0; i < imagesForProcessing.size(); i++) {
      SaveImage(imagesForProcessing[i], "./aligned/testAligned"+to_string(i), ".jpg");
    }
  }

  start = std::chrono::high_resolution_clock::now();
  auto stackedImage = stackImages(imagesForProcessing, base_index, masks, programParams1, stackingParams1);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "BENCHMARK: stacking images"<< " | elapsed time: "<< MicroToSeconds(duration.count()) << " [s]" << std::endl;

  auto[errorMSE_stacked, psnr_stacked] = CompareMetrics(refImage, stackedImage, false);
  start = std::chrono::high_resolution_clock::now();
  auto sharpnessRefImg = sharpnessOfRegions(refImage, 4, 64);
  auto sharpnessBaseImg = sharpnessOfRegions(imagesForProcessing[0], 4, 64);
  auto sharpnessPerRegion = sharpnessOfRegions(MatchResolution(stackedImage, imagesForProcessing[0].size(), INTER_NEAREST), 4, 64);
  stop = std::chrono::high_resolution_clock::now();
  duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
  std::cout << "BENCHMARK: estimating sharpness of image"<< " | elapsed time: "<< MicroToSeconds(duration.count()) << " [s]" << std::endl;
  valarray<float> sharpnessRef = toValarray(sharpnessRefImg);
  float meanSharpRef = sharpnessRef.sum() / (sharpnessRef.size());
  valarray<float> sharpness = toValarray(sharpnessBaseImg);
  float meanSharp = sharpness.sum() / (sharpness.size());
  valarray<float> sharpness2 = toValarray(sharpnessPerRegion);
  float meanSharp2 = sharpness2.sum() / (sharpness2.size());
  auto stop_benchmark = std::chrono::high_resolution_clock::now();
  auto duration_benchmark = std::chrono::duration_cast<std::chrono::microseconds>(stop_benchmark - start_benchmark);
  if(VERBOSITY >= 0) {
    cout<<"BENCHMARK: Input images had misalignment error before prosessing - mse average = "<< mseList_unprocessed.sum()/mseList_unprocessed.size()
        << " | mse per image = " << toVec(mseList_unprocessed) << endl; 
    cout<<"BENCHMARK: Images for stacking were undistorted with result - mse average = "<< mseList.sum()/mseList.size()
        << " | mse per image = " << toVec(mseList) << endl;
    cout<<"BENCHMARK: Average sharpness of reference image = "<< meanSharpRef<<endl;
    cout<<"BENCHMARK: Average sharpness of base image = "<< meanSharp<<endl;
    cout<<"BENCHMARK: Average sharpness of stacked image = "<< meanSharp2<<endl;
    cout<<"BENCHMARK: Similarity to reference of stacked image: mse="<< errorMSE_stacked<< " | psnr="<< psnr_stacked<< endl;
  }
  if(VERBOSITY > 0) {
    SaveImage(stackedImage, "./output/testStacked", ".jpg");
  }
  std::cout << "BENCHMARK: Total elapsed time: "<< MicroToSeconds(duration_benchmark.count()) << " [s]" << std::endl;

  if(meanSharp2 >= meanSharp) {
    result++;
  }

  return result;
}
