from __future__ import annotations

from django.db import models
from api.user.models import User, BaseModel


class Prompt (BaseModel):
    timeframe = models.CharField(max_length=4)
    text = models.TextField()

    def __str__(self):
        return self.text[:10]



class GenData(BaseModel):
    title = models.CharField(max_length=255)
    text = models.TextField()
    tradingview_img_url = models.ImageField(upload_to="images/")
    user = models.ForeignKey(User, related_name="gendatas", on_delete=models.CASCADE)

    def __str__(self):
        return self.title
    
class Report(BaseModel):
    user = models.ForeignKey(User, on_delete=models.CASCADE, related_name="reports")
    report_type = models.CharField(max_length=32)
    symbol = models.CharField(max_length=32)
    result_text = models.TextField()
    created_at = models.DateTimeField(auto_now_add=True)

    def __str__(self):
        return f"{self.user.username} - {self.report_type} - {self.symbol}"
