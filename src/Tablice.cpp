/*
 * Tablice.cpp
 *
 *  Created on: 15 gru 2017
 *      Author: mon
 */

#include "Tablice.h"
#include <iostream>
using namespace cv;


Rect PlateCandidate::rect() {
	int minx = INT32_MAX, miny = INT32_MAX, maxx = 0, maxy = 0;
	for (auto i = ccas.begin(); i != ccas.end(); i++) {
		if (i->x < minx)
			minx = i->x;
		if (i->x + i->w > maxx)
			maxx = i->x + i->w;
		if (i->y < miny)
			miny = i->y;
		if (i->y + i->h > maxy)
			maxy = i->y + i->h;
	}

	minx = max(0.0, minx - 0.2 * h());
	maxx = maxx + 0.2 * h();
	miny = max(0.0, miny - 0.2 * h());
	maxy = maxy + 0.2 * h();

	Rect rect(minx, miny, maxx - minx, maxy - miny);
	return rect;
}





void display_blobs(cv::Mat& labels, std::vector<CCA>& ccas, const char* name) {
	Mat dst(labels.size(), CV_8UC3);
	for (int r = 0; r < labels.rows; r++) {
		for (int c = 0; c < labels.cols; c++) {
			int label = labels.at<int>(r, c);
			Vec3b &pixel = dst.at<Vec3b>(r, c);

			pixel = ccas[label].valid ? ccas[label].color : Vec3b(0, 0, 0);
		}
	}

	imshow(name, dst);

}

void get_matrix(Mat& labels, std::vector<CCA>& ccas, Mat& output) {

	for (int r = 0; r < labels.rows; r++) {
		for (int c = 0; c < labels.cols; c++) {
			int label = labels.at<int>(r, c);
			Vec3b &pixel = output.at<Vec3b>(r, c);

			pixel = ccas[label].valid ? Vec3b(255, 255, 255) : Vec3b(0, 0, 0);
		}
	}
}

void create_cca(Mat& labels, Mat& stats, Mat& centroids,
		std::vector<CCA>& output) {
	for (int i = 0; i < stats.rows; i++) {
		CCA cca;
		cca.label = i;
		cca.x = stats.at<int>(Point(0, i));
		cca.y = stats.at<int>(Point(1, i));
		cca.w = stats.at<int>(Point(2, i));
		cca.h = stats.at<int>(Point(3, i));
		cca.a = stats.at<int>(Point(4, i));
		cca.cx = centroids.at<int>(Point(0,i));
		cca.cy = centroids.at<int>(Point(1,i));
		cca.color = Vec3b(rand() % 256, rand() % 256, rand() % 256);
		if (i > 0)
			cca.valid = true;
		else
			cca.valid = false;
		output.push_back(cca);
	}
}

void filter_cca(std::vector<CCA>& ccas, Mat& image) {
	for (auto i = ccas.begin(); i != ccas.end(); i++) {

		double ar = (double) i->w / i->h;
		double maxwidth = image.size().width / 10.0;
		//area
		if (i->a < 20)
			i->valid = false;
		//width&height
		else if (i->w > maxwidth || i->h > maxwidth * 2.5)
			i->valid = false;
		//aspect ratio
		else if (ar < 1.0 / 9.0 || ar > 3.0 / 4.0)
			i->valid = false;
	}

}

void find_plate_candidates(std::vector<CCA>& ccas,
		std::vector<PlateCandidate>& vec) {
	std::sort(ccas.begin(), ccas.end(), [](const CCA& lhs, const CCA& rhs) {
		if(lhs.x==rhs.x)return lhs.y<rhs.y;
		return lhs.x<rhs.x;
	});

	for (auto i = ccas.begin(); i != ccas.end(); i++) {
		CCA& cca = *i;
		if (cca.valid == false)
			continue;

		auto j = vec.begin();
		for (; j != vec.end(); j++) {
			PlateCandidate& pc = *j;
			if ((abs(cca.y - pc.rightmost_y()) <= 0.1 * pc.h()
					|| abs(cca.y - pc.rightmost_y()) < 5)
					&& cca.x - pc.rightmost_x() <= 1.5*pc.h()
					/*&& pc.rightmost_x() < cca.x*/
					&& ((cca.h <= 1.1 * pc.h() && cca.h >= 0.9 * pc.h())
							|| abs(cca.h - pc.h()) < 5)) {
				pc.ccas.push_back(cca);
				break;
			}
		}
		if (j == vec.end()) {
			vec.push_back(PlateCandidate());
			PlateCandidate& pc = *vec.rbegin();
			pc.ccas.push_back(cca);
		}

	}

	std::sort(ccas.begin(), ccas.end(), [](const CCA& lhs, const CCA& rhs) {
		return lhs.label<rhs.label;
	});
}

void filter_candidates(std::vector<CCA>& ccas,
		std::vector<PlateCandidate>& vec) {
	for (auto i = vec.begin(); i != vec.end(); i++) {
		if (i->ccas.size() > 10 || i->ccas.size() < 6)
			i->valid = 0;

		for (auto j = i->ccas.begin(); j != i->ccas.end(); j++) {
			if (i->valid == 0)
				ccas[j->label].valid = 0;

			ccas[j->label].color = i->color();
		}

	}
}

void display_image_with_rects(Mat& image, std::vector<PlateCandidate>& vec) {
	for (auto i = vec.begin(); i != vec.end(); i++) {
		if (i->valid == 0)
			continue;
		rectangle(image, i->rect(), Vec3b(0, 0, 255), 2);
	}
	imshow("Rects", image);
}

std::string ocr_char(Mat& image, CCA& character,
		tesseract::TessBaseAPI*& tess) {

	tess->TesseractRect(image.data, 1, image.step1(), character.x, character.y,
			character.w, character.h);
	return tess->GetUTF8Text();
}

void ocr(Mat& image, std::vector<PlateCandidate>& vec) {
	tesseract::TessBaseAPI *tess = new tesseract::TessBaseAPI();

	tess->Init(NULL, "eng");
	tess->SetPageSegMode(tesseract::PageSegMode::PSM_SINGLE_LINE);
	for (auto i = vec.begin(); i != vec.end(); i++) {
		if (i->valid == 0)
			continue;
		Rect rect = i->rect();
		tess->TesseractRect(image.data, 3, image.step1(), rect.x, rect.y,
				rect.width, rect.height);
		std::cout << tess->GetUTF8Text() << std::endl;

	}

	tess->Clear();
	tess->End();
}

void set_clahe(Mat& src, Mat& dst) {
	Ptr<CLAHE> clahe = createCLAHE();
	clahe->setClipLimit(16);
	clahe->setTilesGridSize(Size(3, 3));
	clahe->apply(src, dst);
}

void set_unsharp(Mat& src, Mat& dst) {
	Mat blur;
	GaussianBlur(src, blur, Size(7, 7), 5.0);
	addWeighted(src, 6, blur, -5, 0, dst);
}

