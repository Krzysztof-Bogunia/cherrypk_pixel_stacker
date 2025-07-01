#pragma once

// #include "opencv2/core/hal/interface.h"
#include "opencv2/core.hpp"
#include "opencv2/core/hal/interface.h"
#include "opencv2/core/types.hpp"
#include "opencv2/imgproc.hpp"
#include "utils.cpp"
#include "processing.cpp"
#include <cmath>
#include <string>

/**
 * @brief Generate image with list of colors in boxes.
 * @param X list of colors
 * @param Y second list of colors
 * @param sizePx size of square box [px]
 * @return cv::Mat image with plotted list of color boxes
 */
cv::Mat plotColorPairs(const std::vector<std::vector<float>>& X, const std::vector<std::vector<float>>& Y, int sizePx=32) {
  using namespace std;
  using namespace cv;

  //get number of colors
  int N = X[0].size();
  for (int i = 0; i < X.size(); i++) {
    N = min(N, (int)X[i].size());
  }

  Mat imgPlot = Mat::zeros(sizePx*2, sizePx*N, CV_8UC3);

  for (int i = 0; i <N; i++)
  {    
    Scalar color1((uint8_t)X[0][i], (uint8_t)X[1][i], (uint8_t)X[2][i]);
    cv::rectangle(imgPlot, Point2i(i*sizePx,0), Point2i((i+1)*sizePx,sizePx), color1, -1);
  }
  for (int i = 0; i <N; i++)
  {    
    Scalar color2((uint8_t)Y[0][i], (uint8_t)Y[1][i], (uint8_t)Y[2][i]);
    cv::rectangle(imgPlot, Point2i(i*sizePx,sizePx), Point2i((i+1)*sizePx,sizePx+sizePx), color2, -1);
  }

  // cv::cvtColor(imgPlot, imgPlot, cv::COLOR_RGB2BGR);

  return imgPlot;
}

class ColorLUT {
  //TODO: add option for masking
    public:
    std::vector<float> histogramColorRanges = std::vector<float>{0.0f,1.0f};
    int num_dominant_colors = 3; //how many colors to use for alignment 3
    int histSize = 32; // //number or color values per channel //256 //32 //64                     
    float strength = 1.0f; //how much to change/align the color //0.5f 1.0f
    float maxChange = 0.50f; //limit ratio (original*(1+maxChange)) for max color change //0.5f
    int find_colors = num_dominant_colors+2; //how many colors to search for best match //num_dominant_colors+2
    int numChannels = 0;
    cv::Mat lut_8U;    
    cv::Mat lut_32F; // 1d lut with 3 columns (r,g,b) normalized to range histogramColorRanges //[0.0; 1.0]
    cv::Mat lut3d; // 3d lut

    /// @brief Calculates histogram for each channel of input image
    /// @param image multichannel image (BGR, uint8 type)
    /// @param mask optional mask to exclude some regions of image
    /// @return histogram with number of cols equal to channels (RGB); vector of max value per channel ; vector of vectors with indices/bins for each channel of histogram sorted from lest to most significant
    std::tuple<cv::Mat,std::vector<float>,std::vector<std::vector<int>>> calcHistogram(const cv::Mat &image, const cv::Mat &mask=cv::Mat()) {
      using namespace std;
      using namespace cv;
      
      vector<int> channel_id = vector<int>{0};
      vector<int> histSizes = vector<int>{histSize}; 
      std::vector<float> ranges = std::vector<float>{histogramColorRanges[0]*255.0f,histogramColorRanges[1]*255.0f};
      vector<Mat> channels;
      Mat _image = image;
      // cv::cvtColor(_image, _image, cv::COLOR_BGR2RGB);
      split( _image, channels );
      vector<Mat> histInputs(3);
      #pragma omp parallel for
      for (int i = 0; i < channels.size(); i++)
      {
        if(!mask.empty()) {
          Mat mask_ = mask.clone();
          mask_ = invertMask(mask_);
          mask_.convertTo(mask_, CV_8UC1, 255.0);
          calcHist(vector<Mat>{channels[i]},channel_id,mask_,histInputs[i],histSizes, ranges);
        }
        else {
          calcHist(vector<Mat>{channels[i]},channel_id,Mat(),histInputs[i],histSizes, ranges);
        }
      }
  
      Mat histInput = histInputs[0];
      for (int i = 1; i < channels.size(); i++) {
        hconcat(histInput, histInputs[i],histInput);
      }
      histInput.convertTo(histInput, CV_32F);
      vector<float> maxPerChannel(channels.size());
      vector<vector<int>> sortedIndices(channels.size());
      for (int c = 0; c < channels.size(); c++) {
        sortedIndices[c] = Argsort(toVec(histInput.col(c).clone()));
        maxPerChannel[c] = histInput.at<float>(sortedIndices[c][histSize-1], c);
      }
  
      return std::make_tuple(histInput, maxPerChannel, sortedIndices);
    }
  
    /**
     * @brief Compare img2 to reference img1 by calculating histograms.
     * 
     * @param img1 reference image
     * @param img2 target image
     * @param distanceCoeff optional coefficient for distances between sorted bins of histogram
     * @param popularityCoeff optional coefficient for histogram bins popularity(numbers per bin)
     * @return float normalized (0; 1) difference 
     */
    float compare(const cv::Mat& img1, const cv::Mat& img2, float distanceCoeff=1.0f, float popularityCoeff=1.0f) {
      using namespace std;
      using namespace cv;

      auto[img1Hist, img1Max, img1SortedIndices] = calcHistogram(img1);
      auto[img2Hist, img2Max, img2SortedIndices] = calcHistogram(img2);

      float histDelta=0.0f;
      for (int c = 0; c < img1.channels(); c++) {
        valarray<float> coeffs = toValarray(img1Hist.col(c));
        valarray<float> coeffs2 = toValarray(img2Hist.col(c));
        auto topIndices = img1SortedIndices[c];
        auto topIndices2 = img2SortedIndices[c];

        vector<float> dist(topIndices.begin(), topIndices.end());
        valarray<float> distances1 = toValarray(dist);
        dist = vector<float>(topIndices2.begin(), topIndices2.end());
        valarray<float> distances2 = toValarray(dist);
        valarray<float> distances = abs( distances1 - distances2);
        coeffs = NormalizeTo_0_1(coeffs);
        coeffs2 = NormalizeTo_0_1(coeffs2);
        distances = NormalizeTo_0_1(distances);
        valarray<float> totalCoeffs = (popularityCoeff*abs(coeffs - coeffs2) + distanceCoeff*distances) / 2.0f;
        histDelta += totalCoeffs.sum() / totalCoeffs.size();
      }

      return (histDelta / img1.channels()) / (2.0f / (popularityCoeff + distanceCoeff));
    }

    /// @brief Calibrate LUT by comparing series of input images to best matched target images.
    /// @param inputImages vector of images with original look
    /// @param targetImages vector of images with target look
    void calibrate(const std::vector<cv::Mat>& inputImages, const std::vector<cv::Mat>& targetImages) {
      using namespace std;
      using namespace cv;

      numChannels = inputImages[0].channels();
      vector<Mat> histInput(inputImages.size());
      vector<Mat> histTarget(targetImages.size());
      vector<vector<float>> inputMax(inputImages.size());
      vector<vector<float>> targetMax(targetImages.size());
      vector<vector<vector<int>>> inputSortedIndices(inputImages.size());
      vector<vector<vector<int>>> targetSortedIndices(targetImages.size());
      vector<int> imgPairs(inputImages.size(), 0);

      #pragma omp parallel for
      for(int n1=0;n1<inputImages.size();n1++) {
        auto[histInput_, inputMax_, inputSortedIndices_] = calcHistogram(inputImages[n1]);
        histInput[n1] = histInput_;
        inputMax[n1] = inputMax_;
        inputSortedIndices[n1] = inputSortedIndices_;
      }

      #pragma omp parallel for
      for(int n2=0;n2<targetImages.size();n2++) {
        auto[histTarget_, targetMax_, targetSortedIndices_] = calcHistogram(targetImages[n2]);
        histTarget[n2] = histTarget_;
        targetMax[n2] = targetMax_;
        targetSortedIndices[n2] = targetSortedIndices_;
      }

      //find best match
      for(int n1=0; n1<inputImages.size(); n1++) {
        float distance = FLT_MAX;
        for(int n2=0; n2<targetImages.size(); n2++) {
          float totalDiff = this->compare(inputImages[n1], targetImages[n2], 1.0f, 1.0f);
          if(totalDiff < distance) {
            imgPairs[n1] = n2;
            distance = totalDiff;
            if(totalDiff == 0.0f) {
              break;
            }
          }
        }
      }

      valarray<valarray<float>> averageInputColors(valarray<float>(0.0f, num_dominant_colors), numChannels);
      valarray<valarray<float>> averageTargetColors(valarray<float>(0.0f, num_dominant_colors), numChannels);

      for(int n1=0; n1<inputImages.size(); n1++) {
        float histSizeRatio = 255.0f / (float)(histSize-1);
        Mat lut = Mat::zeros(256,numChannels,CV_32F);
        Mat lut2 = Mat::zeros(histSize,numChannels,CV_32F);
        if(VERBOSITY > 0) {
          cout<< "ColorLur::calibrate: matching img "<< n1<< " to "<< imgPairs[n1]<< endl;
        }
        Mat histInputGray, histTargetGray;
        cv::reduce(histInput[n1], histInputGray, 1, REDUCE_SUM, -1);
        cv::reduce(histTarget[imgPairs[n1]], histTargetGray, 1, REDUCE_SUM, -1);
        vector<vector<float>> XX(numChannels);
        vector<vector<float>> channelsColors(numChannels);    

        #pragma omp parallel for
        for (int c = 0; c < numChannels; c++) {
          // valarray<float> coeffs = toValarray(histInput[n1].col(c)) + toValarray(histInputGray);
          // valarray<float> coeffs2 = toValarray(histTarget[imgPairs[n1]].col(c)) + toValarray(histTargetGray);
          valarray<float> coeffs = toValarray(histInput[n1].col(c));
          valarray<float> coeffs2 = toValarray(histTarget[imgPairs[n1]].col(c));
          // valarray<float> coeffsSum = (valarray<float>)(coeffs+coeffs2);
          valarray<float> coeffsSum = (valarray<float>)(toValarray(histInputGray) + toValarray(histTargetGray));
          auto topIndices = Argsort(toVec((valarray<float>)(coeffs)));
          auto topIndices2 = Argsort(toVec((valarray<float>)(coeffs2)));
          // auto topIndices = Argsort(toVec((valarray<float>)(coeffs + toValarray(histInputGray))));
          // auto topIndices2 = Argsort(toVec((valarray<float>)(coeffs2 + toValarray(histTargetGray))));
          topIndices = lastN(topIndices, find_colors);
          topIndices2 = lastN(topIndices2, find_colors);
          // auto topIndices = Argsort(toVec((valarray<float>)(coeffs+coeffsSum)));
          // auto topIndices2 = Argsort(toVec((valarray<float>)(coeffs2+coeffsSum)));
    
          inputSortedIndices[n1][c] = lastN(topIndices, num_dominant_colors);
          targetSortedIndices[imgPairs[n1]][c] = topIndices2;
          vector<float> dist(topIndices.begin(), topIndices.end());
          valarray<float> distances1 = toValarray(dist);
          dist = vector<float>(topIndices2.begin(), topIndices2.end());
          valarray<float> distances2 = toValarray(dist);
          valarray<float> distances = abs( distances2 - distances1 );
          coeffs = select(coeffs, topIndices);
          coeffs2 = select(coeffs2, topIndices2);
          coeffs = NormalizeTo_0_1(coeffs);
          coeffs2 = NormalizeTo_0_1(coeffs2);
          distances = NormalizeTo_0_1(distances);
          float distStrength = 1.0f;
          float coeffsStrength = 1.0f;
          valarray<float> totalCoeffs = coeffsStrength*coeffs2 - distStrength*distances;
          totalCoeffs = NormalizeTo_0_1(totalCoeffs);
          auto alignedIndices = Argsort(toVec( (valarray<float>) ( totalCoeffs )));

          vector<float> X(num_dominant_colors, -1.0f);
          vector<float> scaledColors(num_dominant_colors, -1.0f);
          vector<int> badIndices;
          int iter;
          for (int i = 0; i < num_dominant_colors; i++) {
            iter = alignedIndices.size()-i-1-badIndices.size(); //iterating from end, skip existing values
            // iter = i+badIndices.size(); 
            if(iter < 0) {
              break;
            }
            float refVal = (float)inputSortedIndices[n1][c][num_dominant_colors-i-1];
            auto [targetVal, targetIndex] = closestValue(targetSortedIndices[imgPairs[n1]][c], (float)refVal);
            // remove(targetSortedIndices[imgPairs[n1]][c], targetIndex);
            // targetVal = (targetVal + targetVal + 1.0f) / 2.0f; //the middle of histogram bin
            float distance = (targetVal-refVal) * 255.0f / histSizeRatio;
            float maxChangeScaled = maxChange * 255.0f / histSizeRatio;
            float scaledVal;
            scaledVal = lerp(refVal, targetVal, strength);
            scaledVal = LimitToRange(scaledVal, refVal-maxChangeScaled, refVal+maxChangeScaled); //limit change
            float possibleX = refVal * histSizeRatio;
            X[i] = possibleX;
            scaledColors[i] = scaledVal * histSizeRatio;
            // scaledColors[i] = round(scaledVal * histSizeRatio);
            // //add only unique points
            // if(find(X.begin(), X.end(), possibleX) == X.end()) {
            //   X[i] = possibleX;
            //   scaledColors[i] = round(scaledVal * histSizeRatio);
            // }
            // else {
            //   badIndices.push_back((int)possibleX);
            //   i--;
            // }
          }
    
          auto sortedIndices = Argsort(X);
          X = Reorder(X, sortedIndices);
          scaledColors = Reorder(scaledColors, sortedIndices);
          XX[c] = X;
          channelsColors[c] = scaledColors;
        }

        // //swap colorspace rgb/bgr
        // XX = swap(XX, 0, 2);
        // channelsColors = swap(channelsColors, 0, 2);

        averageInputColors += toValarray(XX);
        averageTargetColors += toValarray(channelsColors);

        // TODO: try 3d interpolation (linear?)
        #pragma omp parallel for
        for (int c = 0; c < numChannels; c++) {
          vector<float> X = XX[c];
          vector<float> Y = channelsColors[c];
          X.push_back((float)0);
          X.push_back(round((float)(255)));
          Y.push_back((float)0);
          Y.push_back(round((float)(255)));
          auto[Xu, indUnique] = unique(X);
          X = Xu;
          // remove(scaledColors, indDuplicate);
          Y = select(Y, indUnique);
          auto sortedIndices2 = Argsort(X);
          X = Reorder(X, sortedIndices2);
          Y = Reorder(Y, sortedIndices2);
          // TODO: interpolators possibly cant extrapolate and fill with 0 when out of range
          _1D::LinearInterpolator<float> interp1; //LinearDelaunayTriangleInterpolator  ThinPlateSplineInterpolator
          _1D::LinearInterpolator<float> interp2;
          interp1.setData(X,Y);
          interp2.setData(X,Y);
          for(int x=0; x<256; x++) {
            float y = interp1((float)x);
            lut.at<float>(x,c) = (float)LimitToRange(round(y), (float)0, (float)255);
          }
          for(int x=0; x<histSize; x++) {
            float y = interp2((float)(x)*histSizeRatio);
            // lut2.at<float>(x,c) = (float)y;
            lut2.at<float>(x,c) = (float)LimitToRange(y, (float)0, (float)255);
          }
        }

        // //swap colorspace RGB/BGR
        // lut = swapCol(lut,0,2);
        // lut2 = swapCol(lut2,0,2);

        lut.convertTo(lut, CV_8U, 1.0/(double)(inputImages.size()));
        lut2.convertTo(lut2, CV_32F, 1.0/(double)(inputImages.size()));
        if(lut_8U.empty()) {
          lut_8U = lut;
        }
        else {
          lut_8U = add(lut_8U, lut);
        }
        if(lut_32F.empty()) {
          lut_32F = lut2;
        }
        else {
          lut_32F = add(lut_32F, lut2);
        }
      }
      lut3d = ColorLUT::from1dto3d(lut_32F, this->histSize, this->histogramColorRanges[0], this->histogramColorRanges[1]);
      averageInputColors = divide(averageInputColors, (float)(inputImages.size()));
      averageTargetColors = divide(averageTargetColors, (float)(inputImages.size()));
    }
 
    /// @brief Apply 1D LUT. Uses linear interpolation for missing values. Works with SRGB 8-bit colorspace.
    static cv::Mat apply(const cv::Mat& img, const cv::Mat& lut) {
      using namespace std;
      using namespace cv;
    
      Mat _image = img.clone();
      // cvtColor(_image, _image, COLOR_BGR2RGB);
      vector<Mat> channels;
      split(_image, channels);
      vector<Mat> calibratedChannels(channels.size());
      //normalize lut to colorspace values
      double minVal, maxVal;
      cv::minMaxLoc(lut, &minVal, &maxVal);
      Mat lutScaled;
      float histogramRatio = 255.0f / (float)(lut.rows - 1);
      lut.convertTo(lutScaled, CV_32F, 1.0, (double)-minVal);
      lutScaled.convertTo(lutScaled, CV_32F, (double)(255.0)/(maxVal-minVal));

      for (int c = 0; c < channels.size(); c++) {
        Mat calibrated = Mat::zeros(img.rows,img.cols,CV_8UC1);
        #pragma omp parallel for
        for (int i = 0; i < img.rows; i++) {
          for (int j = 0; j < img.cols; j++) {
            float current = (float)channels[c].at<uint8_t>(i,j);
            float scaledColor = current / histogramRatio;
            float floorColor, fractionColor;
            fractionColor = std::modf(scaledColor, &floorColor);
            float previousColor = (float)(lutScaled.at<float>((int)(floorColor),c));
            float nextColor = (float)(lutScaled.at<float>((int)LimitToRange(floorColor+1, 0, 255), c));
            float targetColor = lerp(previousColor, nextColor, fractionColor) ;
            calibrated.at<uint8_t>(i,j) = (uint8_t)LimitToRange(targetColor, 0, 255); //limit values to supported range
          }
        }
        calibratedChannels[c] = calibrated;   
      }
      Mat colorCalibrated;
      cv::merge(calibratedChannels, colorCalibrated);
      // cvtColor(colorCalibrated, colorCalibrated, COLOR_RGB2BGR);
      return colorCalibrated;
    }

    /// @brief Apply 1D LUT. Uses trilinear interpolation for missing values. Works with SRGB 8-bit colorspace.
    static cv::Mat apply2(const cv::Mat& img, const cv::Mat& lut) {
      using namespace std;
      using namespace cv;
    
      Mat _image = img;
      cvtColor(_image, _image, COLOR_BGR2RGB);
      vector<Mat> channels;
      vector<Mat> calibratedChannels;
      split(_image, channels);
      split(_image, calibratedChannels);
      //normalize lut to colorspace values
      double minVal, maxVal;
      cv::minMaxLoc(lut, &minVal, &maxVal);
      Mat lutScaled;
      float histogramRatio = 255.0f / (float)(lut.rows - 1);
      lut.convertTo(lutScaled, CV_32F, 1.0, (double)-minVal);
      lutScaled.convertTo(lutScaled, CV_32F, 1.0/(double)(histogramRatio));

      Mat calibrated = Mat::zeros(img.rows,img.cols,CV_8UC1);
      #pragma omp parallel for
      for (int i = 0; i < img.rows; i++) {
        for (int j = 0; j < img.cols; j++) {
          Mat current(1,3,CV_32F);
          for (int c = 0; c < img.channels(); c++) {
            current.at<float>(0,c) = (float)channels[c].at<uint8_t>(i,j);
          }
          Mat scaledColor = divide(current, histogramRatio);
          //Find color cube corners
          valarray<float> V0(img.channels());
          valarray<float> V1(img.channels());
          valarray<float> distance(img.channels());
          for (int c = 0; c < img.channels(); c++) {
            float floorColor, fractionColor;
            fractionColor = std::modf(scaledColor.at<float>(0,c), &floorColor);
            V0[c] = lutScaled.at<float>((int)floorColor, c);
            V1[c] = lutScaled.at<float>((int)LimitToRange(floorColor + 1.0f, 0.0f, (float)(lut.rows-1)), c) ;
            distance[c] = fractionColor * (V1[c] - V0[c]);
          }
          valarray<valarray<float>> Vcube { {V0[0],V0[1],V0[2]}, 
                                        {V0[0],V0[1],V1[2]}, 
                                        {V0[0],V1[1],V0[2]}, 
                                        {V0[0],V1[1],V1[2]}, 
                                        {V1[0],V0[1],V0[2]}, 
                                        {V1[0],V0[1],V1[2]}, 
                                        {V1[0],V1[1],V0[2]}, 
                                        {V1[0],V1[1],V1[2]} };

          //interpolate red
          valarray<valarray<float>> Vr(valarray<float>(3), 4);
          int index = 0;
          for (int g = 0; g < 2; g++) {
            for (int b = 0; b < 2; b++)
            {
              int index0 = 0*2*2 + g*2 + b;
              int index1 = 1*2*2 + g*2 + b;
              Vr[index] = Vcube[index0] * (1.0f - distance[0]) + Vcube[index1] * distance[0];
              index++;
            }
          }
          //interpolate green
          valarray<valarray<float>> Vrg(valarray<float>(3), 2);
          index = 0;
          for (int b = 0; b < 2; b++)
          {
            int index0 = 0*2 + b;
            int index1 = 1*2 + b;
            Vrg[index] = Vr[index0] * (1.0f - distance[1]) + Vr[index1] * distance[1];
            index++;
          }
          //interpolate blue
          valarray<float> Vrgb(3);
          Vrgb = Vrg[0] * (1.0f - distance[2]) + Vrg[1] * distance[2];

          //limit values to supported range
          for (int c = 0; c < img.channels(); c++) {
            float targetColor = Vrgb[c] * histogramRatio;
            // float targetColor = Vrgb[c];
            calibratedChannels[c].at<uint8_t>(i,j) = (uint8_t)round(LimitToRange(targetColor, (float)0.0f, (float)255.0f)); 
          }
        }
      }

      //merge channels
      Mat colorCalibrated;
      cv::merge(calibratedChannels, colorCalibrated);
      cvtColor(colorCalibrated, colorCalibrated, COLOR_RGB2BGR); //swap colorspace
      return colorCalibrated;
    }

    /// @brief Apply 1D LUT. Uses linear interpolation for missing values.
    cv::Mat apply(const cv::Mat& img) {
      using namespace std;
      using namespace cv;
    
      Mat colorCalibrated = ColorLUT::apply(img, this->lut_32F);
      return colorCalibrated;
    }

    /// @brief Apply 3D LUT. Uses linear interpolation for missing values.
    cv::Mat apply2(const cv::Mat& img) {
      using namespace std;
      using namespace cv;
    
      Mat colorCalibrated = ColorLUT::apply2(img, this->lut_32F);
      return colorCalibrated;
    }
    
    static cv::Mat from1dto3d(const cv::Mat& lut2d, int LUT_1D_SIZE, float DOMAIN_MIN, float DOMAIN_MAX) {
      using namespace std;
      using namespace cv;
    
      Mat lutScaled = lut2d.clone();
      double minVal, maxVal;
      cv::minMaxLoc(lutScaled, &minVal, &maxVal);
      if(minVal < (double)DOMAIN_MIN) {
        // lutScaled = add(lut2d, (float)-DOMAIN_MIN);
        lutScaled = add(lutScaled, DOMAIN_MIN-(float)(minVal));
      }
      cv::minMaxLoc(lutScaled, &minVal, &maxVal);
      if(maxVal > (double)DOMAIN_MAX) {
      // lutScaled.convertTo(lutScaled, CV_32F, 1.0/(DOMAIN_MAX-DOMAIN_MIN));
        lutScaled = divide(lutScaled, (float)(maxVal)/DOMAIN_MAX);
      }
      Mat lut3d_ = Mat::zeros(LUT_1D_SIZE*LUT_1D_SIZE*LUT_1D_SIZE,lut2d.cols, CV_32F);
      // average values for each channel (r,g,b)
      int index=0;
      for (int b = 0; b < LUT_1D_SIZE; b++) {
        for (int g = 0; g < LUT_1D_SIZE; g++) {
          for (int r = 0; r < LUT_1D_SIZE; r++) {
            lut3d_.at<float>(index,0) = (float)lutScaled.at<float>(r, 0);
            lut3d_.at<float>(index,1) = (float)lutScaled.at<float>(g, 1);
            lut3d_.at<float>(index,2) = (float)lutScaled.at<float>(b, 2);
            index++;
          }
        }
      }

      return lut3d_;
    }

    static cv::Mat from3dto1d(const cv::Mat& lut3d, int LUT_3D_SIZE, float DOMAIN_MIN, float DOMAIN_MAX) {
      using namespace std;
      using namespace cv;
    
      Mat lutScaled = Mat::zeros(lut3d.rows,lut3d.cols, CV_32F);
      lutScaled = add(lut3d, (float)-DOMAIN_MIN);
      lutScaled.convertTo(lutScaled, CV_32F, 1.0/(DOMAIN_MAX-DOMAIN_MIN));
      Mat lut2d = Mat::zeros(LUT_3D_SIZE,lut3d.cols, CV_32F);
      // average values for each channel (r,g,b)
      int index=0;
      for (int b = 0; b < LUT_3D_SIZE; b++) {
        for (int g = 0; g < LUT_3D_SIZE; g++) {
          for (int r = 0; r < LUT_3D_SIZE; r++) {
            lut2d.at<float>(r, 0) += (float)lutScaled.at<float>(index,0);
            lut2d.at<float>(g, 1) += (float)lutScaled.at<float>(index,1);
            lut2d.at<float>(b, 2) += (float)lutScaled.at<float>(index,2);
            index++;
          }
        }
      }
      // float R_ratio = (float)(LUT_3D_SIZE * LUT_3D_SIZE);
      // float G_rati0 = (float)(LUT_3D_SIZE);
      // float B_rati0 = (float)1.0f;
      float R_ratio = (float)(LUT_3D_SIZE * LUT_3D_SIZE);
      float G_rati0 = (float)(LUT_3D_SIZE * LUT_3D_SIZE);
      float B_rati0 = (float)(LUT_3D_SIZE * LUT_3D_SIZE);
      Mat R_lut = divide(Mat(lut2d.col(0).clone()), R_ratio);
      Mat G_lut = divide(Mat(lut2d.col(1).clone()), G_rati0);
      Mat B_lut = divide(Mat(lut2d.col(2).clone()), B_rati0);
      R_lut.copyTo(lut2d.col(0));
      G_lut.copyTo(lut2d.col(1));
      B_lut.copyTo(lut2d.col(2));

      return lut2d;
    }

    void load(const std::string &filePath) {
      using namespace std;
      using namespace cv;

      if(!filesystem::exists(filePath)) {
        return;
      }
      else {
        std::setlocale(LC_ALL, "en_US.UTF-8"); // use"." as decimal separator
        fstream file;
        string line;
        file.open(filePath, ios::in);
        bool cube3d = false;
        Mat lut;
        int index = 0;
        while(getline(file, line)) {
          vector<string> values = split(line, ' ');
          if((values.size()>0) && (isEqual(values[0], "LUT_3D_SIZE", true))) {
            histSize = stoi(values[1]);
            cube3d = true;
          }
          else if((values.size()>0) && (isEqual(values[0], "LUT_1D_SIZE", true))) {
            histSize = stoi(values[1]);
            cube3d = false;
          }
          else if((values.size()>0) && (isEqual(values[0], "DOMAIN_MIN", true))) {
            histogramColorRanges[0] = stof(values[1]);
          }
          else if((values.size()>0) && (isEqual(values[0], "DOMAIN_MAX", true))) {
            histogramColorRanges[1] = stof(values[1]);
          }
          else if((values.size() == 3) && (!cube3d)) {
            if(index <= 0) {
              lut = Mat::zeros(histSize, 3, CV_32F);
            }
            lut.at<float>(index, 0) = stof(values[0]);
            lut.at<float>(index, 1) = stof(values[1]);
            lut.at<float>(index, 2) = stof(values[2]);
            index++;
          }
          else if(values.size() == 3) {
            if(index <= 0) {
              lut = Mat::zeros(histSize*histSize*histSize, 3, CV_32F);
            }
            lut.at<float>(index, 0) = (float)stof(values[0]);
            lut.at<float>(index, 1) = (float)stof(values[1]);
            lut.at<float>(index, 2) = (float)stof(values[2]);
            index++;
          }
        }
        file.close();
        if(cube3d) {
          lut_32F = from3dto1d(lut, histSize, histogramColorRanges[0], histogramColorRanges[1]);
        }
        else {
          lut_32F = add(lut, (float)-histogramColorRanges[0]);
          lut_32F.convertTo(lut_32F, CV_32F, 1.0/(histogramColorRanges[1]-histogramColorRanges[0]));
        }
        lut_32F.convertTo(lut_8U, CV_8UC3, 255.0);
      }
    }

    /// @brief Saves color look-up table (lut) to 3d cube format. Number of points (r,g,b) in file is lut.rows^3.
    /// @param filePath filePath of file [+date]
    /// @param addTimestamp option to add timestamp in file name
    /// @param overwrite option to overwrite existing file
    void save(const std::string &filePath="./lut", bool addTimestamp=true, bool overwrite=false) {
      using namespace std;
      using namespace cv;
  
      string fileType = ".cube";
      string _filePath = "";
      if(addTimestamp) {
        time_t timestamp = time(NULL);
        struct tm datetime = *localtime(&timestamp);
        char dateFormatted[16];
        strftime(dateFormatted, 16, "%Y%m%d_%H%M%S", &datetime);
        _filePath = filePath+"_"+dateFormatted+"_"+fileType;
      }
      else {
        _filePath = filePath + fileType;
      }

      Mat lutScaled = lut3d.clone();
      double minVal, maxVal;
      cv::minMaxLoc(lutScaled, &minVal, &maxVal);
      if(minVal < (double)histogramColorRanges[0]) {
        lutScaled = add(lutScaled, histogramColorRanges[0]-(float)(minVal));
      }
      cv::minMaxLoc(lutScaled, &minVal, &maxVal);
      if(maxVal > (double)histogramColorRanges[1]) {
        lutScaled = divide(lutScaled, (float)(maxVal)/histogramColorRanges[1]);
      }
      
      vector<double> minVals(lutScaled.cols), maxVals(lutScaled.cols);
      for (int c = 0; c < lutScaled.cols; c++) {
        cv::minMaxLoc(lutScaled.col(c), &minVals[c], &maxVals[c]);
        minVals[c] = round(minVals[c]);
        maxVals[c] = round(maxVals[c]);
      }
  
      //dont overwrite existing file
      if(filesystem::exists(_filePath) && !overwrite) {
        return;
      }
      else {
        fstream file;
        file.open(_filePath, ios::out);
        file << "LUT_3D_SIZE"<< " "<< histSize<< endl;
        file << endl;
        file << std::fixed<< std::setprecision(3);
        file << "DOMAIN_MIN"<<" "<< minVals[0]<<" "<< minVals[1]<<" "<< minVals[2]<< endl;
        file << "DOMAIN_MAX"<<" "<< maxVals[0]<<" "<< maxVals[1]<<" "<< maxVals[2]<< endl;
        file << endl;
        for (int i = 0; i < lutScaled.rows; i++)
        {
          file << (float)lutScaled.at<float>(i,0)<<" "
                <<(float)lutScaled.at<float>(i,1)<<" "
                <<(float)lutScaled.at<float>(i,2)
                << endl;
        }
        file.close();
      }
    }

    /// @brief Saves color look-up table (lut) to 1d cube format. Number of points (r,g,b) in file is lut.rows.
    /// @param filePath filePath of file [+date]
    /// @param addTimestamp option to add timestamp in file name
    /// @param overwrite option to overwrite existing file
    void save1d(const std::string &filePath="lut", bool addTimestamp=true, bool overwrite=false) {
      using namespace std;
      using namespace cv;
  
      string fileType = ".cube";
      string _filePath = "";
      if(addTimestamp) {
        time_t timestamp = time(NULL);
        struct tm datetime = *localtime(&timestamp);
        char dateFormatted[16];
        strftime(dateFormatted, 16, "%Y%m%d_%H%M%S", &datetime);
        _filePath = filePath+"_"+dateFormatted+"_"+fileType;
      }
      else {
        _filePath = filePath + fileType;
      }

      Mat lutScaled = lut_32F.clone();
      double minVal, maxVal;
      cv::minMaxLoc(lutScaled, &minVal, &maxVal);
      if(minVal < (double)histogramColorRanges[0]) {
        lutScaled = add(lutScaled, histogramColorRanges[0]-(float)(minVal));
      }
      cv::minMaxLoc(lutScaled, &minVal, &maxVal);
      if(maxVal > (double)histogramColorRanges[1]) {
        lutScaled = divide(lutScaled, (float)(maxVal)/histogramColorRanges[1]);
      }

      vector<double> minVals(lutScaled.cols), maxVals(lutScaled.cols);
      for (int c = 0; c < lutScaled.cols; c++) {
        cv::minMaxLoc(lutScaled.col(c), &minVals[c], &maxVals[c]);
        minVals[c] = round(minVals[c]);
        maxVals[c] = round(maxVals[c]);
      }
  
      //dont overwrite existing file
      if(filesystem::exists(_filePath) && !overwrite) {
        return;
      }
      else {
        fstream file;
        file.open(_filePath, ios::out);
        file << "LUT_1D_SIZE"<<" "<< histSize<< endl;
        file << endl;
        file << std::fixed<< std::setprecision(3);
        file << "DOMAIN_MIN"<<" "<< minVals[0]<<" "<< minVals[1]<<" "<< minVals[2]<< endl;
        file << "DOMAIN_MAX"<<" "<< maxVals[0]<<" "<< maxVals[1]<<" "<< maxVals[2]<< endl;
        file << endl;
        for (int i = 0; i < lutScaled.rows; i++) {
          file << (float)lutScaled.at<float>(i,0)<<" "
                <<(float)lutScaled.at<float>(i,1)<<" "
                <<(float)lutScaled.at<float>(i,2)
                << endl;
        }
        file.close();
      }
    }

  };
  

