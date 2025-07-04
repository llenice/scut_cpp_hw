/********************************************************************************
** Form generated from reading UI file 'widget.ui'
**
** Created by: Qt User Interface Compiler version 6.9.1
**
** WARNING! All changes made in this file will be lost when recompiling UI file!
********************************************************************************/

#ifndef UI_WIDGET_H
#define UI_WIDGET_H

#include <QtCore/QVariant>
#include <QtWidgets/QApplication>
#include <QtWidgets/QGridLayout>
#include <QtWidgets/QLineEdit>
#include <QtWidgets/QPushButton>
#include <QtWidgets/QWidget>

QT_BEGIN_NAMESPACE

class Ui_Widget
{
public:
    QLineEdit *mainLineEdit;
    QWidget *layoutWidget;
    QGridLayout *gridLayout;
    QPushButton *SUB;
    QPushButton *THREE;
    QPushButton *NINE;
    QPushButton *FOUR;
    QPushButton *EQUAL;
    QPushButton *ZEOR;
    QPushButton *TIMES;
    QPushButton *SEVEN;
    QPushButton *RIGHT;
    QPushButton *PLUS;
    QPushButton *DELETE;
    QPushButton *EIGHT;
    QPushButton *TWO;
    QPushButton *ONE;
    QPushButton *LEFT;
    QPushButton *CLEAR;
    QPushButton *DIVIDE;
    QPushButton *FIVE;
    QPushButton *SIX;
    QPushButton *DIVIDE_2;

    void setupUi(QWidget *Widget)
    {
        if (Widget->objectName().isEmpty())
            Widget->setObjectName("Widget");
        Widget->resize(335, 295);
        mainLineEdit = new QLineEdit(Widget);
        mainLineEdit->setObjectName("mainLineEdit");
        mainLineEdit->setGeometry(QRect(70, 20, 191, 20));
        layoutWidget = new QWidget(Widget);
        layoutWidget->setObjectName("layoutWidget");
        layoutWidget->setGeometry(QRect(60, 50, 214, 218));
        gridLayout = new QGridLayout(layoutWidget);
        gridLayout->setObjectName("gridLayout");
        gridLayout->setContentsMargins(0, 0, 0, 0);
        SUB = new QPushButton(layoutWidget);
        SUB->setObjectName("SUB");
        QSizePolicy sizePolicy(QSizePolicy::Policy::Fixed, QSizePolicy::Policy::Fixed);
        sizePolicy.setHorizontalStretch(0);
        sizePolicy.setVerticalStretch(0);
        sizePolicy.setHeightForWidth(SUB->sizePolicy().hasHeightForWidth());
        SUB->setSizePolicy(sizePolicy);
        SUB->setMinimumSize(QSize(40, 40));
        SUB->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(SUB, 0, 2, 1, 1);

        THREE = new QPushButton(layoutWidget);
        THREE->setObjectName("THREE");
        sizePolicy.setHeightForWidth(THREE->sizePolicy().hasHeightForWidth());
        THREE->setSizePolicy(sizePolicy);
        THREE->setMinimumSize(QSize(40, 40));
        THREE->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(THREE, 3, 2, 1, 1);

        NINE = new QPushButton(layoutWidget);
        NINE->setObjectName("NINE");
        sizePolicy.setHeightForWidth(NINE->sizePolicy().hasHeightForWidth());
        NINE->setSizePolicy(sizePolicy);
        NINE->setMinimumSize(QSize(40, 40));
        NINE->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(NINE, 1, 2, 1, 1);

        FOUR = new QPushButton(layoutWidget);
        FOUR->setObjectName("FOUR");
        sizePolicy.setHeightForWidth(FOUR->sizePolicy().hasHeightForWidth());
        FOUR->setSizePolicy(sizePolicy);
        FOUR->setMinimumSize(QSize(40, 40));
        FOUR->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(FOUR, 2, 0, 1, 1);

        EQUAL = new QPushButton(layoutWidget);
        EQUAL->setObjectName("EQUAL");
        QSizePolicy sizePolicy1(QSizePolicy::Policy::Minimum, QSizePolicy::Policy::Maximum);
        sizePolicy1.setHorizontalStretch(0);
        sizePolicy1.setVerticalStretch(0);
        sizePolicy1.setHeightForWidth(EQUAL->sizePolicy().hasHeightForWidth());
        EQUAL->setSizePolicy(sizePolicy1);
        EQUAL->setMinimumSize(QSize(40, 40));
        EQUAL->setMaximumSize(QSize(40, 80));

        gridLayout->addWidget(EQUAL, 4, 3, 1, 1);

        ZEOR = new QPushButton(layoutWidget);
        ZEOR->setObjectName("ZEOR");
        sizePolicy.setHeightForWidth(ZEOR->sizePolicy().hasHeightForWidth());
        ZEOR->setSizePolicy(sizePolicy);
        ZEOR->setMinimumSize(QSize(40, 40));
        ZEOR->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(ZEOR, 4, 1, 1, 1);

        TIMES = new QPushButton(layoutWidget);
        TIMES->setObjectName("TIMES");
        sizePolicy.setHeightForWidth(TIMES->sizePolicy().hasHeightForWidth());
        TIMES->setSizePolicy(sizePolicy);
        TIMES->setMinimumSize(QSize(40, 40));
        TIMES->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(TIMES, 1, 3, 1, 1);

        SEVEN = new QPushButton(layoutWidget);
        SEVEN->setObjectName("SEVEN");
        sizePolicy.setHeightForWidth(SEVEN->sizePolicy().hasHeightForWidth());
        SEVEN->setSizePolicy(sizePolicy);
        SEVEN->setMinimumSize(QSize(40, 40));
        SEVEN->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(SEVEN, 1, 0, 1, 1);

        RIGHT = new QPushButton(layoutWidget);
        RIGHT->setObjectName("RIGHT");
        sizePolicy.setHeightForWidth(RIGHT->sizePolicy().hasHeightForWidth());
        RIGHT->setSizePolicy(sizePolicy);
        RIGHT->setMinimumSize(QSize(40, 40));
        RIGHT->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(RIGHT, 4, 2, 1, 1);

        PLUS = new QPushButton(layoutWidget);
        PLUS->setObjectName("PLUS");
        sizePolicy.setHeightForWidth(PLUS->sizePolicy().hasHeightForWidth());
        PLUS->setSizePolicy(sizePolicy);
        PLUS->setMinimumSize(QSize(40, 40));
        PLUS->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(PLUS, 0, 1, 1, 1);

        DELETE = new QPushButton(layoutWidget);
        DELETE->setObjectName("DELETE");
        sizePolicy.setHeightForWidth(DELETE->sizePolicy().hasHeightForWidth());
        DELETE->setSizePolicy(sizePolicy);
        DELETE->setMinimumSize(QSize(40, 40));
        DELETE->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(DELETE, 0, 3, 1, 1);

        EIGHT = new QPushButton(layoutWidget);
        EIGHT->setObjectName("EIGHT");
        sizePolicy.setHeightForWidth(EIGHT->sizePolicy().hasHeightForWidth());
        EIGHT->setSizePolicy(sizePolicy);
        EIGHT->setMinimumSize(QSize(40, 40));
        EIGHT->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(EIGHT, 1, 1, 1, 1);

        TWO = new QPushButton(layoutWidget);
        TWO->setObjectName("TWO");
        sizePolicy.setHeightForWidth(TWO->sizePolicy().hasHeightForWidth());
        TWO->setSizePolicy(sizePolicy);
        TWO->setMinimumSize(QSize(40, 40));
        TWO->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(TWO, 3, 1, 1, 1);

        ONE = new QPushButton(layoutWidget);
        ONE->setObjectName("ONE");
        sizePolicy.setHeightForWidth(ONE->sizePolicy().hasHeightForWidth());
        ONE->setSizePolicy(sizePolicy);
        ONE->setMinimumSize(QSize(40, 40));
        ONE->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(ONE, 3, 0, 1, 1);

        LEFT = new QPushButton(layoutWidget);
        LEFT->setObjectName("LEFT");
        sizePolicy.setHeightForWidth(LEFT->sizePolicy().hasHeightForWidth());
        LEFT->setSizePolicy(sizePolicy);
        LEFT->setMinimumSize(QSize(40, 40));
        LEFT->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(LEFT, 4, 0, 1, 1);

        CLEAR = new QPushButton(layoutWidget);
        CLEAR->setObjectName("CLEAR");
        sizePolicy.setHeightForWidth(CLEAR->sizePolicy().hasHeightForWidth());
        CLEAR->setSizePolicy(sizePolicy);
        CLEAR->setMinimumSize(QSize(40, 40));
        CLEAR->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(CLEAR, 0, 0, 1, 1);

        DIVIDE = new QPushButton(layoutWidget);
        DIVIDE->setObjectName("DIVIDE");
        sizePolicy.setHeightForWidth(DIVIDE->sizePolicy().hasHeightForWidth());
        DIVIDE->setSizePolicy(sizePolicy);
        DIVIDE->setMinimumSize(QSize(40, 40));
        DIVIDE->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(DIVIDE, 2, 3, 1, 1);

        FIVE = new QPushButton(layoutWidget);
        FIVE->setObjectName("FIVE");
        sizePolicy.setHeightForWidth(FIVE->sizePolicy().hasHeightForWidth());
        FIVE->setSizePolicy(sizePolicy);
        FIVE->setMinimumSize(QSize(40, 40));
        FIVE->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(FIVE, 2, 1, 1, 1);

        SIX = new QPushButton(layoutWidget);
        SIX->setObjectName("SIX");
        sizePolicy.setHeightForWidth(SIX->sizePolicy().hasHeightForWidth());
        SIX->setSizePolicy(sizePolicy);
        SIX->setMinimumSize(QSize(40, 40));
        SIX->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(SIX, 2, 2, 1, 1);

        DIVIDE_2 = new QPushButton(layoutWidget);
        DIVIDE_2->setObjectName("DIVIDE_2");
        sizePolicy.setHeightForWidth(DIVIDE_2->sizePolicy().hasHeightForWidth());
        DIVIDE_2->setSizePolicy(sizePolicy);
        DIVIDE_2->setMinimumSize(QSize(40, 40));
        DIVIDE_2->setMaximumSize(QSize(40, 40));

        gridLayout->addWidget(DIVIDE_2, 3, 3, 1, 1);


        retranslateUi(Widget);

        QMetaObject::connectSlotsByName(Widget);
    } // setupUi

    void retranslateUi(QWidget *Widget)
    {
        Widget->setWindowTitle(QCoreApplication::translate("Widget", "Widget", nullptr));
        SUB->setText(QCoreApplication::translate("Widget", "-", nullptr));
        THREE->setText(QCoreApplication::translate("Widget", "3", nullptr));
        NINE->setText(QCoreApplication::translate("Widget", "9", nullptr));
        FOUR->setText(QCoreApplication::translate("Widget", "4", nullptr));
        EQUAL->setText(QCoreApplication::translate("Widget", "=", nullptr));
        ZEOR->setText(QCoreApplication::translate("Widget", "0", nullptr));
        TIMES->setText(QCoreApplication::translate("Widget", "*", nullptr));
        SEVEN->setText(QCoreApplication::translate("Widget", "7", nullptr));
        RIGHT->setText(QCoreApplication::translate("Widget", "\357\274\211", nullptr));
        PLUS->setText(QCoreApplication::translate("Widget", "+", nullptr));
        DELETE->setText(QString());
        EIGHT->setText(QCoreApplication::translate("Widget", "8", nullptr));
        TWO->setText(QCoreApplication::translate("Widget", "2", nullptr));
        ONE->setText(QCoreApplication::translate("Widget", "1", nullptr));
        LEFT->setText(QCoreApplication::translate("Widget", "\357\274\210", nullptr));
        CLEAR->setText(QCoreApplication::translate("Widget", "AC", nullptr));
        DIVIDE->setText(QCoreApplication::translate("Widget", "/", nullptr));
        FIVE->setText(QCoreApplication::translate("Widget", "5", nullptr));
        SIX->setText(QCoreApplication::translate("Widget", "6", nullptr));
        DIVIDE_2->setText(QCoreApplication::translate("Widget", ".", nullptr));
    } // retranslateUi

};

namespace Ui {
    class Widget: public Ui_Widget {};
} // namespace Ui

QT_END_NAMESPACE

#endif // UI_WIDGET_H
