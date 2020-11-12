from django.apps import AppConfig
import quickstart.RID.RID.main


class MyAppConfig(AppConfig):
    name = 'quickstart'
    verbose_name = "My App"

    def ready(self):
        print('start')
        app.run(main)
        pass