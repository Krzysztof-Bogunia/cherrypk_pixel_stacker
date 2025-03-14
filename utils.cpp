#include <iostream>
#include <fstream>
#include <string>
#include <json/json.h>
#include <json/value.h>
#include <opencv2/core.hpp>
// #include <opencv2/opencv.hpp>
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


// Overloaded operators
//  Override printing vector
template <typename S>
std::ostream &operator<<(std::ostream &os,
                         const std::vector<S> &vector)
{
  for (auto element : vector)
  {
    os << element << " ";
  }
  return os;
}
template<typename T>
bool isEqual(const std::valarray<T> &left, const std::valarray<T> &right) {
    return std::equal(std::begin(left), std::end(left), std::begin(right));
}


//** Data conversion functions **
std::vector<float> toVec(Json::Value jval)
{
  using namespace std;
  vector<float> result;
  for (int i = 0; i < jval.size(); i++)
  {
    result.push_back(jval[i].asFloat());
  }
  return result;
}

std::vector<std::vector<float>> toVec2D(Json::Value jval)
{
  using namespace std;
  vector<vector<float>> result;
  for (int i = 0; i < jval.size(); i++)
  {
    vector<float> result2 = toVec(jval[i]);
    result.push_back(result2);
  }
  return result;
}

template<typename T>
std::valarray<T> toValarray(std::vector<T> vec){
  std::valarray<T> varr(vec.data(), vec.size()); //std::vector to valarray
  return varr;
}

template<typename T>
std::valarray<std::valarray<T>> toValarray(std::vector<std::vector<T>> vec){
  std::valarray<std::valarray<T>> varr;
  varr.resize(vec.size());
  for(int i=0;i<vec.size();i++) {
      std::valarray<T> varr2(vec[i].data(), vec[i].size()); //std::vector to valarray
      varr[i]=varr2;
  }
  return varr;
}

cv::Mat toMat(std::vector<float> vec)
{
  using namespace std;

  // cv::Mat newMat(0, vec.size(), cv::DataType<float>::type);
  // newMat.push_back(vec.data());
  cv::Mat newMat(1, vec.size(), cv::DataType<float>::type, vec.data());

  return newMat;
}

cv::Mat toMat(std::vector<std::vector<float>> vec)
{
  using namespace std;
  cv::Mat newMat(0, vec[0].size(), cv::DataType<float>::type);
  for (unsigned int i = 0; i < vec.size(); ++i)
  {
    cv::Mat mat1d(1, vec[i].size(), cv::DataType<float>::type, vec[i].data());
    newMat.push_back(mat1d);
  }
  return newMat;
}

cv::Mat toMat(std::vector<std::vector<double>> vec)
{
  using namespace std;
  cv::Mat newMat(0, vec[0].size(), cv::DataType<double>::type);
  for (unsigned int i = 0; i < vec.size(); ++i)
  {
    cv::Mat mat1d(1, vec[i].size(), cv::DataType<double>::type, vec[i].data());
    newMat.push_back(mat1d);
  }
  return newMat;
}

cv::Mat toMat(const std::vector<cv::Point2f> &vecPoints2D)
{
  using namespace std;
  using namespace cv;

  int rows = vecPoints2D.size();
  Mat result = Mat::zeros(rows, 2, CV_32F);
  for (int i = 0; i < rows; i++)
  {
    result.at<float>(i, 0) = vecPoints2D[i].x;
    result.at<float>(i, 1) = vecPoints2D[i].y;
  }
  return result;
}

cv::Mat to3dMat(const std::vector<cv::Mat>& vec){
  using namespace cv;
  int sizes[3] = {(int)vec.size(), vec[0].cols, vec[0].rows};
  cv::Mat matCombined;
  cv::merge(vec, matCombined);

  return matCombined;
}

std::vector<float> toVec(const cv::Mat& m){
  std::vector<float> vec(m.begin<float>(), m.end<float>());
  return vec;
}
template<typename T>
std::vector<T> toVec(const cv::Mat& m, T matType=(float)0.0f){
  std::vector<T> vec(m.begin<T>(), m.end<T>());
  return vec;
}

template<typename T>
std::vector<std::vector<T>> toVec2D(const cv::Mat& m, T matType=(float)0.0f){
  std::vector<std::vector<T>> vec;
  for(int r=0; r<m.rows; r++){
    cv::Mat row = m.row(r);
    std::vector<T> vec_row(row.begin<T>(), row.end<T>());
    vec.push_back(vec_row);
  }
  return vec;
}

std::valarray<float> toValarray(const cv::Mat& m){
  std::vector<float> vec = toVec(m);
  std::valarray<float> varr = toValarray(vec);
  return varr;
}

std::vector<cv::Mat> toVecMat(const cv::Mat& mat3D){
  using namespace cv;
  std::vector<cv::Mat> result;
  cv::split(mat3D, result);
  return result;
}

double MicroToSeconds(int64_t time)
{
  return (double)time / 1000000.0;
}

template<typename T>
std::vector<T> toVec(const std::valarray<T>& varr){
  std::vector<T> vec(varr.size());
  std::copy(std::begin(varr), std::end(varr), std::begin(vec));
  return vec;
}

std::vector<cv::Point2f> toVecPoint2f(const cv::Mat &arrayPoints) {
//Convert each row to point. Each column is considered separate dimension of points.
  std::vector<cv::Point2f> imagePoints(arrayPoints.rows); // 2d points in new image
  for(int i=0; i < arrayPoints.rows; ++i) {
      imagePoints.at(i).x = arrayPoints.at<float>(i, 0);
      imagePoints.at(i).y = arrayPoints.at<float>(i, 1);
  }
  return imagePoints;
}

std::vector<cv::Point2f> toVecPoint2f(const std::vector<cv::Point3f> &arrayPoints) {
  std::vector<cv::Point2f> imagePoints(arrayPoints.size());
  for(int i=0; i < arrayPoints.size(); ++i) {
      imagePoints.at(i).x = arrayPoints.at(i).x;
      imagePoints.at(i).y = arrayPoints.at(i).y;
  }
  return imagePoints;
}

std::vector<cv::Point3f> toVecPoint3f(const cv::Mat &arrayPoints) {
//Convert each row to point. Each column is considered separate dimension of points.
  std::vector<cv::Point3f> imagePoints(arrayPoints.rows); // 3d points in new image
  if(arrayPoints.cols >= 3) {
    for(int i=0; i < arrayPoints.rows; ++i) {
      imagePoints.at(i).x = arrayPoints.at<float>(i,0);
      imagePoints.at(i).y = arrayPoints.at<float>(i,1);
      imagePoints.at(i).z = arrayPoints.at<float>(i,2);
    }
  }
  else {
    for(int i=0; i < arrayPoints.rows; ++i) {
      imagePoints.at(i).x = arrayPoints.at<float>(i,0);
      imagePoints.at(i).y = arrayPoints.at<float>(i,1);
      imagePoints.at(i).z = 0.0f;
    }
  }
  return imagePoints;
}

std::string to_string(const std::list<cv::Point> &data) {
  using namespace std;

  string result = "";
  int i = 0;
  for (cv::Point p : data) {
    if(i>0) { result += ","; }
    auto x = p.x;
    auto y = p.y;
    result = result + "(" + to_string(p.x) + "," + to_string(p.y) + ")";
    i += 1;
  }
  return result;
}


