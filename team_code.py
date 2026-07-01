#!/usr/bin/env python

from lstm_team import load_lstm_model, predict_record, train_lstm_model


def train_model(data_folder, model_folder, verbose):
    train_lstm_model(data_folder, model_folder, verbose)


def load_model(model_folder, verbose):
    return load_lstm_model(model_folder, verbose)


def run_model(model, record, data_folder, verbose):
    return predict_record(model, record, data_folder)
