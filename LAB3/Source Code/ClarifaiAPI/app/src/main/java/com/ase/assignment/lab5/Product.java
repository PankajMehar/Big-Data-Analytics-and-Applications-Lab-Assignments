package com.ase.assignment.lab5;

import android.app.Activity;
import android.app.Dialog;
import android.os.Bundle;
import android.view.View;
import android.view.Window;
import android.widget.Button;
import android.widget.TextView;

import java.util.ArrayList;
import java.util.List;

import clarifai2.dto.prediction.Concept;

/**
 * Created by Shreya on 9/29/2017.
 */

public class Product extends Dialog implements
        View.OnClickListener{
    public Activity c;
    public Dialog d;
    public Button yes, no;
    public List<Concept> cc;
    public Product(Activity a, List<Concept> cc) {
        super(a);
        // TODO Auto-generated constructor stub
        this.c = a;
        this.cc=cc;
    }

    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        requestWindowFeature(Window.FEATURE_NO_TITLE);
        setContentView(R.layout.activity_output);
        yes = (Button) findViewById(R.id.btn_ok);

        yes.setOnClickListener(this);

        ArrayList<TextView> textViewList =new ArrayList<TextView>();
        textViewList.add((TextView)findViewById(R.id.add1));
        textViewList.add((TextView)findViewById(R.id.add2));
        textViewList.add((TextView)findViewById(R.id.add3));
        textViewList.add((TextView)findViewById(R.id.add4));
        textViewList.add((TextView)findViewById(R.id.add5));
        textViewList.get(0).setText("Product"+"\t"+"Value");
        int i=0;
        for(Concept concept:cc){
            if(i>4)break;
            textViewList.get(i).setText(concept.name()+"\t"+concept.value());

            i++;
        }

    }

    @Override
    public void onClick(View v) {
        switch (v.getId()) {
            case R.id.btn_ok:
                this.dismiss();
                break;

            default:
                break;
        }

    }

}
