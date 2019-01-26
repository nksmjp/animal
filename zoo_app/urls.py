urlpatterns = [
    path('', views.classify, name='classify'),
    path('history', views.history, name='history')
    path('signup', views.signup, name='signup')