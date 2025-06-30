#pragma once

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
#include <cctype>
#include <opencv2/imgproc.hpp>
#include "opencv2/highgui.hpp"
#include "opencv2/features2d.hpp"
#include "opencv2/video.hpp"
#include "opencv2/calib3d.hpp"
#include <limits>
#include <unordered_set>
#include <ctime>
#include "opencv2/photo.hpp"

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

bool isEqual(const std::string& left, const std::string& right, bool caseInsensitive=false) {
  if (left.size() != right.size()) {
      return false;
  }
  if (caseInsensitive) {
    for (int i=0; i<left.size(); i++) {
      if (tolower((unsigned char)left[i]) != tolower((unsigned char)right[i])) {
        return false;
      }
    }
  }
  else {
    for (int i=0; i<left.size(); i++) {
      if (((unsigned char)left[i]) != ((unsigned char)right[i])) {
        return false;
      }
    }
  }
  return true;
}

// Helper function to compare if 2 input values are equal
bool isEqual(const cv::Point2f& p1, const cv::Point2f& p2) {
  return ((p1.x == p2.x) && (p1.y == p2.y));
}

// Helper function to compare if 2 input values are equal
template <class T>
bool isEqual(const std::vector<T>& p1, const std::vector<T>& p2) {
 if(p1.size() != p2.size()) {
   return false;
 }
 for(int i=0;i<p1.size();i++) {
   if(p1[i] == p2[i]) {
     return true;
   }
 }
 return false;
}

// Helper function to compare if 2 input values are equal
template <class T>
bool isEqual(const T& p1, const T& p2) {
 return p1 == p2;
}

// Helper function to compare if 2 input Mats are equal
bool isEqual(const cv::Mat& p1, const cv::Mat& p2) {
 cv::Mat diff = p1 != p2;
 bool eq = cv::sum(diff) == cv::Scalar(0,0,0,0);
  return eq;
}

/**
 * @return true if value2 is greater than value1 
 */
bool isGreater(const std::string value1, const std::string value2) {
  using namespace std;
  
  int result = value1.compare(value2);
  if(result < 0) {
    return true;
  }
  else {
    return false;
  }
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

template<typename T>
std::vector<std::vector<T>> toVec(std::valarray<std::valarray<T>> varr){
  std::vector<std::vector<T>> vec;
  vec.resize(varr.size());
  for(int i=0;i<varr.size();i++) {
      std::vector<T> vec2 = toVec(varr[i]);
      vec[i]=vec2;
  }
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

std::vector<std::string> split(std::string s, char delimiter) {
  using namespace std;

  string word; 
  stringstream ss(s);
  vector<string> words;
  while(getline(ss, word, delimiter)){
      words.push_back(word);
  }

  return words;
}

/// @brief Return N first elements
/// @tparam T 
/// @param s vector of type T
/// @param n number of elements to return
/// @param offset number of elements to skip
/// @return vector of same type with n elements
template<typename T>
std::vector<T> firstN(const std::vector<T>& vec, int n, int offset=0) {
  using namespace std;

  int n2 = min(n, (int)vec.size());
  vector<T> vec2(n2);
  for (int i = 0; i < n2; i++)
  {
    vec2[i] = vec[i+offset];
  }  

  return vec2;
}

/// @brief Return N last elements
/// @tparam T 
/// @param s vector of type T
/// @param n number of elements to return
/// @param offset number of elements to skip
/// @return vector of same type with n elements
template<typename T>
std::vector<T> lastN(const std::vector<T>& vec, int n, int offset=0) {
  using namespace std;

  int n2 = min(n, (int)vec.size());
  vector<T> vec2(n2);
  for (int i = 0; i < n2; i++)
  {
    vec2[n2-1-i] = vec[vec.size()-1-i-offset];
  }  

  return vec2;
}

