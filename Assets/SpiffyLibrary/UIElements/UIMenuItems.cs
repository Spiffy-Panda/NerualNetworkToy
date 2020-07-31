using System.Collections;
using System.Collections.Generic;
using System.Collections.Specialized;
using System.Linq;
using System.Net.Http.Headers;
using System.Text;
using System.Xml;
using UnityEditor;
using UnityEngine;
using UnityEngine.UIElements;

namespace SpiffyLibrary.UIElements
{
  public class UIMenuItems 
  {
    [MenuItem("SpiffyLibrary/Scan File For Names")]
    public static void ScanFileForNames() {
      string FileTemplate = "namespace ProjectUI.Generated\n{{\n  " +
                            "public static class {0}\n  {{\n    " +
                            "public static class Names\n    {{\n      " +
                            "{1}\n    }}\n  }}\n}}";
      string NameVarLineTemplate = "public const string {0} = \"{0}\";";
      Debug.Log(Selection.activeObject.name);
      if (!(Selection.activeObject is VisualTreeAsset))
        return;
      string filename = AssetDatabase.GetAssetPath(Selection.instanceIDs[0]);
      var doc = new XmlDocument();
      doc.Load(filename);
      List<string> elementNames = new List<string>();
      void DepthFirstSearch(XmlNode node) {
        string elmName = node.Attributes["name"]?.Value;
        if(elmName != null)
          elementNames.Add(elmName);
        for (int iChild = 0; iChild < node.ChildNodes.Count; iChild++) {
          DepthFirstSearch(node.ChildNodes[iChild]);
        }
      }
      DepthFirstSearch( doc.FirstChild);
      elementNames.Sort();
      string content = string.Join("\n      ", elementNames.Select(str => string.Format(NameVarLineTemplate, str)));
      string className= System.IO.Path.GetFileNameWithoutExtension(filename) + "Names";
      string fileContent = (string.Format(FileTemplate,className, content));
      System.IO.File.WriteAllText(filename.Split('.')[0] + "Names.gen.cs", fileContent );
    }

  }

}
