#include <iostream>
#include <vector>
#include <string>
#include <filesystem>

#include <opencv2/opencv.hpp>
#include <opencv2/features2d.hpp>


namespace fs = std::filesystem;

// adjust brightness of img1 to match img2
// matching mean brightness in LAB's L channel
bool adjustBrightness(const cv::Mat& darkImg, const cv::Mat& brightImg, cv::Mat& resultImg) {
    if (darkImg.empty() || brightImg.empty()) {
        std::cerr << "Error: Input image(s) for brightness adjustment are empty." << std::endl;
        return false;
    }

    cv::Mat darkLab, brightLab;
    cv::cvtColor(darkImg, darkLab, cv::COLOR_BGR2Lab);
    cv::cvtColor(brightImg, brightLab, cv::COLOR_BGR2Lab);

    std::vector<cv::Mat> darkChannels(3), brightChannels(3);
    cv::split(darkLab, darkChannels);
    cv::split(brightLab, brightChannels);

    // Calculate mean of L channel for both images
    cv::Scalar meanDarkL = cv::mean(darkChannels[0]);
    cv::Scalar meanBrightL = cv::mean(brightChannels[0]);

    double diffL = meanBrightL[0] - meanDarkL[0];

    // Adjust L channel of the dark image
    darkChannels[0] += diffL;
    // Ensure values are within [0, 255] range (though L can go up to 100, OpenCV scales it to 255)
    cv::threshold(darkChannels[0], darkChannels[0], 255, 255, cv::THRESH_TRUNC); // Cap at 255
    cv::threshold(darkChannels[0], darkChannels[0], 0, 0, cv::THRESH_TOZERO);   // Floor at 0

    cv::Mat adjustedLab;
    cv::merge(darkChannels, adjustedLab);
    cv::cvtColor(adjustedLab, resultImg, cv::COLOR_Lab2BGR);

    std::cout << "Brightness adjustment: L_dark_mean=" << meanDarkL[0]
              << ", L_bright_mean=" << meanBrightL[0]
              << ", applied_diff=" << diffL << std::endl;

    return true;
}


int main() {
    fs::path imageDir = "./images";
    fs::path leftImagePath = imageDir / "left_photo.JPG";
    fs::path rightImagePath = imageDir / "right_photo.JPG";
    fs::path outputImagePath = "./panorama_output.jpg";

    std::cout << "Loading images..." << std::endl;
    cv::Mat imgLeft = cv::imread(leftImagePath.string(), cv::IMREAD_COLOR);
    cv::Mat imgRight = cv::imread(rightImagePath.string(), cv::IMREAD_COLOR);

    std::cout << "Left image loaded: " << std::endl;
    std::cout << "Right image loaded: " << std::endl;

    std::cout << "\nAdjusting brightness..." << std::endl;
    cv::Mat imgLeftAdjusted;
    if (adjustBrightness(imgLeft, imgRight, imgLeftAdjusted)) {
        std::cout << "Brightness of left image adjusted." << std::endl;
    } else {
        std::cerr << "Error: Brightness adjustment failed." << std::endl;
        exit(1);
    }


    std::cout << "\nDetecting features and computing homography..." << std::endl;
    
    // Use AKAZE for feature detection
    cv::Ptr<cv::AKAZE> akaze = cv::AKAZE::create();
    std::vector<cv::KeyPoint> keypointsLeft, keypointsRight;
    cv::Mat descriptorsLeft, descriptorsRight;

    // Detect and compute in parallel (OpenCV might do this internally for compute)
    // For explicit parallelism, you could use std::async or std::thread here
    akaze->detectAndCompute(imgLeftAdjusted, cv::noArray(), keypointsLeft, descriptorsLeft);
    akaze->detectAndCompute(imgRight, cv::noArray(), keypointsRight, descriptorsRight);
    std::cout << "Keypoints detected: Left=" << keypointsLeft.size() << ", Right=" << keypointsRight.size() << std::endl;

    if (keypointsLeft.empty() || keypointsRight.empty()) {
        std::cerr << "Error: No keypoints detected in one or both images." << std::endl;
        return -1;
    }
    if (descriptorsLeft.empty() || descriptorsRight.empty()) {
        std::cerr << "Error: No descriptors computed for one or both images." << std::endl;
        return -1;
    }
     // Ensure descriptors are CV_8U for BFMatcher with Hamming distance (common for AKAZE)
    if (descriptorsLeft.type() != CV_8U) {
        descriptorsLeft.convertTo(descriptorsLeft, CV_8U);
    }
    if (descriptorsRight.type() != CV_8U) {
        descriptorsRight.convertTo(descriptorsRight, CV_8U);
    }


    // Match descriptors using Brute-Force Matcher
    cv::BFMatcher matcher(cv::NORM_HAMMING); // NORM_HAMMING for binary descriptors like AKAZE
    std::vector<std::vector<cv::DMatch>> knn_matches;
    matcher.knnMatch(descriptorsRight, descriptorsLeft, knn_matches, 2); // Find 2 best matches for each descriptor in imgRight

    // Filter matches using Lowe's ratio test
    const float ratio_thresh = 0.75f;
    std::vector<cv::DMatch> good_matches;
    for (size_t i = 0; i < knn_matches.size(); i++) {
        if (knn_matches[i].size() == 2 && knn_matches[i][0].distance < ratio_thresh * knn_matches[i][1].distance) {
            good_matches.push_back(knn_matches[i][0]);
        }
    }
    std::cout << "Good matches found: " << good_matches.size() << std::endl;

    // Optional: Draw matches
    // cv::Mat img_matches;
    // cv::drawMatches(imgRight, keypointsRight, imgLeftAdjusted, keypointsLeft, good_matches, img_matches, cv::Scalar::all(-1),
    //                  cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    // cv::imwrite("./matches.jpg", img_matches);


    if (good_matches.size() < 4) { // Need at least 4 points for homography
        std::cerr << "Error: Not enough good matches to compute homography (need at least 4)." << std::endl;
        return -1;
    }

    std::vector<cv::Point2f> ptsRight, ptsLeft;
    for (size_t i = 0; i < good_matches.size(); i++) {
        ptsRight.push_back(keypointsRight[good_matches[i].queryIdx].pt);
        ptsLeft.push_back(keypointsLeft[good_matches[i].trainIdx].pt);
    }

    // Homography H maps points from Right Image to Left Image's coordinate system
    cv::Mat H = cv::findHomography(ptsRight, ptsLeft, cv::RANSAC, 5.0);
    if (H.empty()) {
        std::cerr << "Error: Homography matrix is empty. Could not compute." << std::endl;
        return -1;
    }
    std::cout << "Homography computed." << std::endl;

    // --- 4. Warp Images and Create Panorama ---
    // The goal is to stitch imgRight to the right of imgLeftAdjusted
    std::cout << "\nStep 4: Warping images and creating panorama..." << std::endl;

    // Calculate the dimensions of the panorama
    // Transform corners of imgRight to imgLeft's coordinate system
    std::vector<cv::Point2f> cornersRight(4);
    cornersRight[0] = cv::Point2f(0, 0);
    cornersRight[1] = cv::Point2f((float)imgRight.cols, 0);
    cornersRight[2] = cv::Point2f((float)imgRight.cols, (float)imgRight.rows);
    cornersRight[3] = cv::Point2f(0, (float)imgRight.rows);
    std::vector<cv::Point2f> cornersRightTransformed(4);
    cv::perspectiveTransform(cornersRight, cornersRightTransformed, H);

    // Determine bounding box for the panorama
    float min_x = 0, max_x = (float)imgLeftAdjusted.cols;
    float min_y = 0, max_y = (float)imgLeftAdjusted.rows;

    for (const auto& pt : cornersRightTransformed) {
        min_x = std::min(min_x, pt.x);
        max_x = std::max(max_x, pt.x);
        min_y = std::min(min_y, pt.y);
        max_y = std::max(max_y, pt.y);
    }

    // Create an offset transformation matrix to ensure all coordinates are positive
    cv::Mat T = cv::Mat::eye(3, 3, CV_64F);
    if (min_x < 0) T.at<double>(0, 2) = -min_x; // Translate X
    if (min_y < 0) T.at<double>(1, 2) = -min_y; // Translate Y

    int panoWidth = static_cast<int>(max_x - min_x);
    int panoHeight = static_cast<int>(max_y - min_y);

    std::cout << "Panorama dimensions: " << panoWidth << "x" << panoHeight << std::endl;
    if (panoWidth <=0 || panoHeight <=0 || panoWidth > 20000 || panoHeight > 10000) { // Safety check
        std::cerr << "Error: Invalid panorama dimensions calculated. " 
                  << "W: " << panoWidth << ", H: " << panoHeight << std::endl;
        std::cerr << "Min X: " << min_x << ", Max X: " << max_x << std::endl;
        std::cerr << "Min Y: " << min_y << ", Max Y: " << max_y << std::endl;
        std::cerr << "Homography might be incorrect." << std::endl;
        return -1;
    }


    // Create the panorama canvas (initialized to black)
    cv::Mat panorama(panoHeight, panoWidth, imgLeftAdjusted.type(), cv::Scalar(0,0,0));

    // Warp left image (adjusted) onto the panorama canvas using the translation T
    cv::Mat warpedLeft;
    cv::warpPerspective(imgLeftAdjusted, warpedLeft, T, cv::Size(panoWidth, panoHeight), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    // cv::imwrite("./images/warped_left_temp.jpg", warpedLeft); // Optional: save intermediate

    // Warp right image onto the panorama canvas using T * H
    cv::Mat warpedRight;
    cv::warpPerspective(imgRight, warpedRight, T * H, cv::Size(panoWidth, panoHeight), cv::INTER_LINEAR, cv::BORDER_CONSTANT, cv::Scalar(0,0,0));
    // cv::imwrite("./images/warped_right_temp.jpg", warpedRight); // Optional: save intermediate


    // --- 5. Merge Images (Remove Overlap from one Image) ---
    std::cout << "\nStep 5: Merging images (left image takes precedence in overlap)..." << std::endl;
    
    // First, copy the warpedLeft image to the panorama. It serves as the base.
    warpedLeft.copyTo(panorama);

    // Create a mask for where warpedLeft has content (is not black from warping)
    cv::Mat maskLeft;
    cv::cvtColor(warpedLeft, maskLeft, cv::COLOR_BGR2GRAY);
    cv::threshold(maskLeft, maskLeft, 0, 255, cv::THRESH_BINARY); // If pixel > 0, it's content

    // Invert the mask: we want to copy warpedRight pixels only where warpedLeft had NO content.
    cv::Mat maskForRight;
    cv::bitwise_not(maskLeft, maskForRight);
    
    // Now copy warpedRight to panorama, but only where the maskForRight is non-zero.
    // This fulfills "delete the overlapping area from one photo" (conceptually, from warpedRight).
    // Any pixel in warpedRight that would overlap with an existing pixel from warpedLeft is NOT copied.
    warpedRight.copyTo(panorama, maskForRight);

    // Alternative, slightly more robust way to combine (pixel by pixel)
    // This also gives left image precedence and avoids issues if warpedLeft had actual black pixels.
    // It's slower but more explicit for the "left takes precedence" logic.
    /*
    for (int r = 0; r < panorama.rows; ++r) {
        for (int c = 0; c < panorama.cols; ++c) {
            cv::Vec3b pixelLeft = warpedLeft.at<cv::Vec3b>(r, c);
            // Check if the pixel in warpedLeft is from the actual image or background from warping
            // A simple check is if it's not pure black. A more robust way is to create a proper mask
            // during warping or by checking if (T * original_point) lands here.
            // For simplicity, if pixelLeft is not black, use it.
            bool leftHasContent = (pixelLeft[0] != 0 || pixelLeft[1] != 0 || pixelLeft[2] != 0);
            
            if (leftHasContent) {
                panorama.at<cv::Vec3b>(r, c) = pixelLeft;
            } else {
                // If left image has no content here, take from right image
                panorama.at<cv::Vec3b>(r, c) = warpedRight.at<cv::Vec3b>(r, c);
            }
        }
    }
    */
    // The maskLeft approach above is generally better and faster.

    std::cout << "Images merged." << std::endl;

    // --- 6. Save Resulting Panorama ---
    std::cout << "\nStep 6: Saving panorama..." << std::endl;
    bool success = cv::imwrite(outputImagePath.string(), panorama);
    if (success) {
        std::cout << "Panorama saved successfully to " << fs::absolute(outputImagePath) << std::endl;
    } else {
        std::cerr << "Error: Could not save panorama image to " << fs::absolute(outputImagePath) << std::endl;
        return -1;
    }

    std::cout << "\nProcessing complete." << std::endl;

    return 0;
}