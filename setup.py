from distutils.core import setup

setup(
      name='SceneSense',
      version='1.0',
      description='Scene Sense', #TODO @Alec add more description
      author='Alec Reed, Brendan Crowe, Lorin Achey',
      author_email='alec.reed@colorado.edu',
      url='',
      packages=['SceneSense', "SceneSense.utils", 'SceneSense.spot_data_processing', "SceneSense.training",
                "SceneSense.visualizations", "SceneSense.plotting"],
)