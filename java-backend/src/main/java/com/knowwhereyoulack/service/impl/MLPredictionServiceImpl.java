package com.knowwhereyoulack.service.impl;

import org.springframework.stereotype.Service;

import com.knowwhereyoulack.dto.WeaknessAnalysisResponse;
import com.knowwhereyoulack.service.MLPredictionService;

@Service
public class MLPredictionServiceImpl implements MLPredictionService {

    @Override
    public WeaknessAnalysisResponse predictWeakness(Long userId) {
        //note currently returns mock data. Replace with Python API or ONNX
        WeaknessAnalysisResponse response = new WeaknessAnalysisResponse();
        response.setTopicName("DSA");
        response.setWeaknessLevel("Weak");
        response.setAccuracyPercentage(45.0);
        return response;
    }
}