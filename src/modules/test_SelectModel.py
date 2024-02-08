import pytest
from machine_learning_models import select_model


class TestSelectModel:
  def test_case1(self):
    model = select_model("Logistic")
    assert hasattr(model, "predict")
  
  def test_case2(self):
    model = select_model("RandomForest")
    assert hasattr(model, "predict")
    
  def test_case3(self):
    model = select_model("SVM")
    assert hasattr(model, "predict")
  
  def test_case4(self):
    assert select_model("www") == "error"
  