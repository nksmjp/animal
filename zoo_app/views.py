from django.shortcuts import render
from django.core.files.storage import FileSystemStorage
from django.conf import settings

# Create your views here.
def classify(request):
    if request.method == 'POST' and request.FILES['predict_img']:
        predict_img = request.FILES['predict_img']
        fs = FileSystemStorage()
        filename = fs.save(predict_img.name, predict_img)
        uploaded_file_url = fs.url(filename)
    return render(request, 'zoo_app/classify.html', {})