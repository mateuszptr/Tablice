/*
 * Tablice.h
 *
 *  Created on: 15 gru 2017
 *      Author: mon
 */

#ifndef SRC_TABLICE_H_
#define SRC_TABLICE_H_

#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <tesseract/baseapi.h>

struct CCA {
	int label;
	int x, y;
	int w, h;
	int a;
	int cx, cy;
	cv::Vec3b color;
	bool valid;
};


struct PlateCandidate {
	std::vector<CCA> ccas;
	bool valid = 1;

	cv::Vec3b color() {
		return first().color;
	}
	int rightmost_x() {
		return last().x + last().w;
	}
	int rightmost_y() {
		return last().y;
	}
	int w() {
		return first().w;
	}
	int h() {
		return first().h;
	}
	CCA& first() {
		return *ccas.begin();
	}
	CCA& last() {
		return *ccas.rbegin();
	}
	cv::Rect rect();
};

void display_blobs(cv::Mat& labels, std::vector<CCA>& ccas, const char* name);
void get_matrix(cv::Mat& labels, std::vector<CCA>& ccas, cv::Mat& output);
void create_cca(cv::Mat& labels, cv::Mat& stats, cv::Mat& centroids,
		std::vector<CCA>& output);
void filter_cca(std::vector<CCA>& ccas, cv::Mat& image);
void find_plate_candidates(std::vector<CCA>& ccas,
		std::vector<PlateCandidate>& vec);
void filter_candidates(std::vector<CCA>& ccas,
		std::vector<PlateCandidate>& vec);
void display_image_with_rects(cv::Mat& image, std::vector<PlateCandidate>& vec);
std::string ocr_char(cv::Mat& image, CCA& character,
		tesseract::TessBaseAPI*& tess);
void ocr(cv::Mat& image, std::vector<PlateCandidate>& vec);
void set_clahe(cv::Mat& src, cv::Mat& dst);
void set_unsharp(cv::Mat& src, cv::Mat& dst);

#endif /* SRC_TABLICE_H_ */
