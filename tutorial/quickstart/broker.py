from django.apps import AppConfig
import quickstart.RID.RID.main
import time

class MyAppConfig(AppConfig):
    name = 'quickstart'
    verbose_name = "My App"

    def ready(self):
        print('start')
        while(True):
            try:
                start = time.time()
                while(True):
                    if(time.time()-start>30):
                        app.run(main)
                        start = time.time()
            except SystemExit:
                pass
        
        pass