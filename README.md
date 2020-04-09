# optical_flow
visualize_optical_flow braucht vier Parameter beim starten.

  video_dir: Den Pfad zu dem Video das bearbeitet werden soll
  
  images_path: Der Pfad in dem die Ergebnisse gespeichert werden
  
  begin_shot,end shot: Damit werden die Grenzen des zu bearbeitenden stück in Millisekunden angegeben
  
Der Output sind zwei Arten von Bildern. Einmal die Frames von denen der optical Flow berechnet wird (genannt source_TIMESTAMP_IN_MS.jpeg)
und der Optical Flow dargestellt als Bild im HSV-Farbraum (flow_TIMESTAMP_IN_MS.jpeg). Dabei ist der Hue der Winkel an dem Pixel (rot 0°, grün 120°, blau 240°)
und der Value die Länge des Vektors.
